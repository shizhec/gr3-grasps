"""Bounded request/reply, with a fresh socket after every timeout."""
from pathlib import Path
import secrets
import traceback

import zmq

from .protocol import VERSION, MAX_BYTES, pack, unpack, identifier, write_json


class Client:
    def __init__(self, endpoint, token, timeout_s=300):
        self.endpoint, self.token, self.timeout_s = endpoint, token, timeout_s

    def call(self, action, **payload):
        context = zmq.Context.instance()
        with context.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.MAXMSGSIZE, MAX_BYTES)
            socket.setsockopt(zmq.SNDTIMEO, int(self.timeout_s*1000))
            socket.setsockopt(zmq.RCVTIMEO, int(self.timeout_s*1000))
            socket.connect(self.endpoint)
            try:
                socket.send(pack(dict(version=VERSION, token=self.token, action=action, **payload)))
                result = unpack(socket.recv())
            except zmq.Again as error:
                raise TimeoutError(f"{action} timed out; no motion was requested") from error
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Server rejected request"))
        return result


class Service:
    def __init__(self, planner, token, output):
        if len(token) < 16:
            raise ValueError("Use a token with at least 16 characters")
        self.planner, self.token, self.output = planner, token, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)

    def handle(self, message):
        if message.get("version") != VERSION or not secrets.compare_digest(str(message.get("token", "")), self.token):
            raise ValueError("Authentication or protocol version rejected")
        if message["action"] == "health":
            return dict(ok=True, version=VERSION, planner=self.planner.name, hardware_control=False)
        if message["action"] == "plan":
            observation = message["observation"]
            key = identifier(observation["observation_id"])
            directory = self.output / key
            directory.mkdir(exist_ok=False)  # No implicit repeat of a task.
            plan = self.planner.plan(observation, directory)
            write_json(directory / "plan.json", plan)
            return dict(ok=True, plan=plan)
        if message["action"] == "report":
            report = message["report"]
            directory = self.output / identifier(report["observation_id"])
            if not directory.is_dir():
                raise ValueError("Unknown observation")
            import json
            planned = json.loads((directory/"plan.json").read_text())
            if report.get("task_id") != planned["task_id"]:
                raise ValueError("Report task does not match the planned task")
            write_json(directory / "robot_report.json", report)
            return dict(ok=True)
        raise ValueError("Unsupported action")

    def serve(self, endpoint, stop=None, ready=None):
        with zmq.Context.instance().socket(zmq.REP) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.MAXMSGSIZE, MAX_BYTES)
            socket.bind(endpoint)
            if ready:
                ready.set()
            print(f"Inference service listening on {endpoint}; planner={self.planner.name}", flush=True)
            while stop is None or not stop.is_set():
                if not socket.poll(100):
                    continue
                try:
                    response = self.handle(unpack(socket.recv()))
                except Exception as error:
                    traceback.print_exc()
                    response = dict(ok=False, error=f"{type(error).__name__}: {error}")
                socket.send(pack(response))
