import threading

import msgpack
import msgpack_numpy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import zmq

from gr3_grasping.geometry import transform
from gr3_grasping.grasps import GraspGen


def test_official_wire_protocol_and_base_to_contact_transform():
    context = zmq.Context()
    server = context.socket(zmq.REP)
    port = server.bind_to_random_port("tcp://127.0.0.1")
    pose = transform(translation=[0.02, -0.03, 0.7]).astype(np.float32)
    received = []

    def serve():
        try:
            for _ in range(2):
                request = msgpack.unpackb(server.recv(), object_hook=msgpack_numpy.decode, raw=False)
                received.append(request)
                if request["action"] == "metadata":
                    response = {"gripper_name": "franka_panda", "model_name": "test_protocol_only"}
                else:
                    response = {"grasps": pose[None], "confidences": np.array([0.75], dtype=np.float32)}
                server.send(msgpack.packb(response, default=msgpack_numpy.encode, use_bin_type=True))
        finally:
            server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    client = GraspGen({"host": "127.0.0.1", "port": port, "timeout_ms": 1000,
                       "num_grasps": 20, "topk_num_grasps": 5})
    try:
        T_world_camera = transform(Rotation.from_euler("y", 30, degrees=True).as_matrix(), [0.2, 0, 1.4])
        cloud = np.ones((32, 3), dtype=np.float32) * 0.1
        result = client.infer(cloud, T_world_camera)
        expected = T_world_camera @ pose @ transform(translation=[0, 0, 0.10527314])
        np.testing.assert_allclose(result[0].T_world_tcp, expected)
        assert result[0].score == pytest.approx(0.75)
        np.testing.assert_equal(received[1]["point_cloud"], cloud)
        assert received[1]["num_grasps"] == 20
    finally:
        client.close()
        thread.join(timeout=2)
        context.term()


def test_unavailable_server_fails_with_bounded_timeout():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    socket.close()
    context.term()
    with pytest.raises(TimeoutError):
        GraspGen({"host": "127.0.0.1", "port": port, "timeout_ms": 50})
