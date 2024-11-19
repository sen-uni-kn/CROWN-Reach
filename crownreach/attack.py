from typing import Callable

import torch
import onnxruntime as ort
import numpy as np
import math
import yaml
from torchdiffeq import odeint

from .crown import resolve_resource

ort.set_default_logger_severity(4)


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def compile_dynamics(dynamics_expressions):
    dynamics = [expr.replace("^", "**") for expr in dynamics_expressions]
    return [compile(expr, "<string>", "eval") for expr in dynamics]


def get_dtype_from_onnx_model(session):
    output_info = session.get_outputs()[0]
    if output_info.type == "tensor(float)":
        return torch.float32
    elif output_info.type == "tensor(double)":
        return torch.float64
    return torch.float32


def load_onnx_model(model_path, config) -> tuple[Callable, torch.dtype]:
    ort_session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    dtype = get_dtype_from_onnx_model(ort_session)

    def run(y):
        y = torch.atleast_2d(y)
        input_vars = (
            y[:, : config["num_nn_input"]]
            .detach()
            .numpy()
            .astype(np.float64 if dtype == torch.float64 else np.float32)
        )
        u = ort_session.run(None, {input_name: input_vars})[0]
        return torch.tensor(u, dtype=dtype)

    return run, dtype


def simulate(controller, dynamics, initial_conditions: torch.Tensor, config: dict):
    def dy(t, y):
        locals_dict = {
            config["initial_set"][i]["name"]: y[:, i] for i in range(y.shape[1])
        }

        control_outputs = controller(y)
        control_outputs = control_outputs.expand(-1, config["num_nn_output"])
        adjusted_control_outputs = (
            control_outputs * config["output_scale"] + config["output_offset"]
        )
        for i in range(config["num_nn_output"]):
            locals_dict[f"u{i + 1}"] = adjusted_control_outputs[:, i]

        safe_builtins = {
            "abs": torch.abs,
            "min": torch.minimum,
            "max": torch.maximum,
            "sin": torch.sin,
            "cos": torch.cos,
        }
        dydt = [
            eval(expr, {"__builtins__": safe_builtins}, locals_dict)
            for expr in dynamics
        ]
        for i in range(len(dydt)):
            if not isinstance(dydt[i], torch.Tensor):
                dydt[i] = torch.tensor(dydt[i], dtype=y.dtype).expand(y.shape[0])
        return torch.stack(dydt, dim=-1)

    total_time = config["steps"] * config["step_size"]
    times = torch.linspace(0, total_time, steps=int(config["steps"]))
    initial_conditions = torch.atleast_2d(initial_conditions)
    trajectory = odeint(dy, initial_conditions, times)
    return trajectory.permute(1, 0, 2)  # (batch, time, state)


def evaluate_constraints(trajectory, constraints, config):
    values = torch.empty(*trajectory.shape[:2], len(constraints))
    is_sat = torch.empty(*trajectory.shape[:2], len(constraints), dtype=torch.bool)
    for c, constraint in enumerate(constraints):
        check_expr = compile(constraint.replace("^", "**"), "<string>", "eval")
        for t in range(trajectory.shape[1]):
            locals_dict = {
                config["initial_set"][i]["name"]: trajectory[:, t, i]
                for i in range(trajectory.shape[2])
            }
            vals = eval(check_expr, {"__builtins__": {}}, locals_dict)
            values[:, t, c] = vals
            is_sat[:, t, c] = torch.le(vals, 0)
    return is_sat, values


def check(trajectory, config):
    """Checks target, unsafe and safe constraints.

    Returns a boolean satisfaction vector and a satisfaction value tensor.
    """
    result = torch.full(trajectory.shape[:1], True, dtype=torch.bool)
    value = torch.full(trajectory.shape[:1], torch.inf)
    if "constraints_target" in config:
        is_sat, val = evaluate_constraints(
            trajectory[-1:], config["constraints_target"], config
        )
        result = result & is_sat.all(dim=(1, 2))
        value = torch.minimum(value, val.amin(dim=-1).amin(dim=-1))

    if "constraints_unsafe" in config:
        is_sat, val = evaluate_constraints(
            trajectory, config["constraints_unsafe"], config
        )
        result = result & ~is_sat.all(dim=(1, 2))
        value = torch.minimum(value, -val.amax(dim=-1).amax(dim=-1))

    if "constraints_safe" in config:
        is_sat, val = evaluate_constraints(
            trajectory, config["constraints_safe"], config
        )
        result = result & is_sat.all(dim=(1, 2))
        value = torch.minimum(value, val.amin(dim=-1).amin(dim=-1))
    return result, value


def attack(config_path):
    config = load_config(config_path)
    if not config["run_attack"]:
        return None, None

    model_path = resolve_resource(config["model_dir"], config_path)
    controller, dtype = load_onnx_model(model_path, config)
    dynamics = compile_dynamics(config["dynamics_expressions"])

    initial_set = config["initial_set"]
    initial_conditions = [
        np.random.uniform(low=var["interval"][0], high=var["interval"][1])
        for var in initial_set
    ]
    initial_conditions = torch.tensor(initial_conditions, dtype=dtype)

    trajectory = simulate(controller, dynamics, initial_conditions, config)
    falsified = not check(trajectory, config)[0].all()
    return falsified, trajectory


if __name__ == "__main__":
    try:
        attack("configs/attitude_control.yaml")
    except Exception as e:
        print("results unknown, please use verifier.")
