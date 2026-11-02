"""ONNX wrapper for the start rule's arithmetic. NOT a trained network: there are no learned weights.

The graph maps the Ising coefficients (h of shape [n], J of shape [n, n]) to the k starting parameter
vectors (shape [k, 4]):  scale = sqrt(sum h^2 + sum_{i<j} J_ij^2);  gamma = c_gamma / scale;  beta = c_beta.
It uses Trilu to take the strict upper triangle of J, so n is dynamic. Float32 throughout.

What it covers: ONLY this static arithmetic. The optimization procedure (optimize.py) is an iterative loop
that calls the quantum backend and cannot be part of an ONNX graph. For a function this small, plain Python
is simpler; ONNX is useful only if you want the same artifact format and runtime as your other served models.

    python -m qaoa_start_rule.onnx_export --out qaoa_start_rule.onnx        # writes and verifies the model
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

if __package__ in (None, ""):
    raise SystemExit(
        "Run this as a module, from the folder that CONTAINS the qaoa_start_rule folder (not from inside it):\n"
        "    python -m qaoa_start_rule.onnx_export --out qaoa_start_rule.onnx\n"
        "or, after `pip install -e .`, from anywhere:  qaoa-start-rule-onnx --out qaoa_start_rule.onnx")

from .rule import CONSTANT_STARTS, LAYERS, RULE_VERSION, suggest_starts

OPSET = 17


def build_model(k: int = 4):
    try:
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as e:
        raise ImportError("ONNX support needs:  pip install onnx onnxruntime") from e
    c = CONSTANT_STARTS[:k].astype(np.float32)
    inits = [numpy_helper.from_array(np.ascontiguousarray(c[:, :LAYERS]), "c_gamma"),
             numpy_helper.from_array(np.ascontiguousarray(c[:, LAYERS:]), "c_beta"),
             numpy_helper.from_array(np.array(1, dtype=np.int64), "diagonal_offset")]
    nodes = [
        helper.make_node("Mul", ["h", "h"], ["h_squared"]),
        helper.make_node("ReduceSum", ["h_squared"], ["sum_h"], keepdims=0),
        helper.make_node("Trilu", ["J", "diagonal_offset"], ["J_upper"], upper=1),
        helper.make_node("Mul", ["J_upper", "J_upper"], ["J_squared"]),
        helper.make_node("ReduceSum", ["J_squared"], ["sum_J"], keepdims=0),
        helper.make_node("Add", ["sum_h", "sum_J"], ["total"]),
        helper.make_node("Sqrt", ["total"], ["scale"]),
        helper.make_node("Div", ["c_gamma", "scale"], ["gamma"]),
        helper.make_node("Concat", ["gamma", "c_beta"], ["starts"], axis=1),
    ]
    graph = helper.make_graph(
        nodes, "qaoa_start_rule",
        inputs=[helper.make_tensor_value_info("h", TensorProto.FLOAT, ["n"]),
                helper.make_tensor_value_info("J", TensorProto.FLOAT, ["n", "n"])],
        outputs=[helper.make_tensor_value_info("starts", TensorProto.FLOAT, [k, 2 * LAYERS])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="qaoa_start_rule")
    model.ir_version = 8                          # widely supported by onnxruntime versions
    for key, val in (("rule_version", RULE_VERSION),
                     ("nature", "arithmetic rule, not a trained network"),
                     ("validated_for", "6 assets, budget 3, p=2"),
                     ("parameter_order", "gamma_1, gamma_2, beta_1, beta_2")):
        entry = model.metadata_props.add()
        entry.key, entry.value = key, val
    return model


def export(path: str, k: int = 4) -> str:
    import onnx
    model = build_model(k)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def verify(path: str, k: int = 4, trials: int = 20, seed: int = 0) -> float:
    """Run the model with onnxruntime on random problems (several sizes) and compare with the numpy rule.
    Returns the largest relative difference."""
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("verifying needs:  pip install onnxruntime") from e
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    worst = 0.0
    for t in range(trials):
        n = [4, 6, 6, 8][t % 4]
        h = rng.normal(-0.1, 0.06, n)
        J = np.triu(rng.normal(0.03, 0.02, (n, n)), 1)
        (out,) = sess.run(None, {"h": h.astype(np.float32), "J": J.astype(np.float32)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = suggest_starts(h, J, k)
        worst = max(worst, float(np.max(np.abs(out - ref) / np.maximum(np.abs(ref), 1e-6))))
    return worst


def main():
    ap = argparse.ArgumentParser(description="Export and verify the ONNX start rule")
    ap.add_argument("--out", default="qaoa_start_rule.onnx")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    export(args.out, args.k)
    worst = verify(args.out, args.k)
    print(f"wrote {args.out}; largest relative difference to the numpy rule over random problems: {worst:.2e}")
    if worst > 1e-5:
        raise SystemExit("verification failed")


if __name__ == "__main__":
    main()
