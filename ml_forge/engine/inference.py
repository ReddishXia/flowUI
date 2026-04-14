"""
inference.py
Inference execution and result window.

The dedicated Inference tab is the primary source of truth:
Inference dataset/preprocess chain -> checkpoint node -> InferenceOutput.
"""

from __future__ import annotations

import pathlib
import random

import dearpygui.dearpygui as dpg

import ml_forge.state as state
from ml_forge.ui.console import log


_TEX_TAG = "inf_preview_tex"
_PREV_W = 224
_PREV_H = 224

_CHECKPOINT_NODE_LABELS = ("pth", "pt", "ckpt")
_OUTPUT_NODE_LABEL = "InferenceOutput"
_INF_DATASET_LABELS = (
    "Inf MNIST",
    "Inf FashionMNIST",
    "Inf CIFAR10",
    "Inf CIFAR100",
    "Inf ImageFolder",
)
_INF_AUG_LABELS = (
    "Inf Resize",
    "Inf CenterCrop",
    "Inf Normalize",
    "Inf ToTensor",
    "Inf Grayscale",
)

_state: dict = {
    "ckpt_path": "",
    "last_sample": None,
    "source": "manual",
}


def _ensure_texture() -> None:
    if dpg.does_item_exist(_TEX_TAG):
        return
    blank = [0.1, 0.1, 0.1, 1.0] * (_PREV_W * _PREV_H)
    with dpg.texture_registry():
        dpg.add_dynamic_texture(_PREV_W, _PREV_H, blank, tag=_TEX_TAG)


def _pil_to_texture(pil_img) -> None:
    import numpy as np

    img = pil_img.convert("RGBA").resize((_PREV_W, _PREV_H))
    data = (np.array(img, dtype=np.float32) / 255.0).flatten().tolist()
    dpg.set_value(_TEX_TAG, data)


def _resolve_user_path(path: str) -> str:
    path = path.strip()
    if not path:
        return ""
    p = pathlib.Path(path)
    if p.is_absolute():
        return str(p)
    if state.current_file:
        return str((pathlib.Path(state.current_file).parent / p).resolve())
    return str(p)


def _safe_int(raw, default=5) -> int:
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _ordered_tabs_for_inference() -> list[tuple[int, dict]]:
    ordered: list[tuple[int, dict]] = []
    seen: set[int] = set()

    for tid, tab in state.tabs.items():
        if tab.get("role") == "inference":
            ordered.append((tid, tab))
            seen.add(tid)

    if state.active_tab_id in state.tabs and state.active_tab_id not in seen:
        ordered.append((state.active_tab_id, state.tabs[state.active_tab_id]))
        seen.add(state.active_tab_id)

    for tid, tab in state.tabs.items():
        if tid not in seen:
            ordered.append((tid, tab))
            seen.add(tid)

    return ordered


def _iter_checkpoint_nodes() -> list[dict]:
    from ml_forge.graph.nodes import input_field_tag

    nodes: list[dict] = []
    for tid, tab in _ordered_tabs_for_inference():
        for ntag, node_info in tab["nodes"].items():
            label = node_info["label"] if isinstance(node_info, dict) else str(node_info)
            if label not in _CHECKPOINT_NODE_LABELS:
                continue
            nid = int(ntag.split("_")[2])
            ftag = input_field_tag(tid, nid, "path")
            path = dpg.get_value(ftag).strip() if dpg.does_item_exist(ftag) else ""
            nodes.append({
                "tid": tid,
                "tab": tab,
                "ntag": ntag,
                "label": label,
                "path": path,
                "field_tag": ftag,
            })
    return nodes


def _iter_output_nodes() -> list[dict]:
    from ml_forge.graph.nodes import input_field_tag

    nodes: list[dict] = []
    for tid, tab in _ordered_tabs_for_inference():
        for ntag, node_info in tab["nodes"].items():
            label = node_info["label"] if isinstance(node_info, dict) else str(node_info)
            if label != _OUTPUT_NODE_LABEL:
                continue
            nid = int(ntag.split("_")[2])
            ftag = input_field_tag(tid, nid, "top_k")
            top_k = dpg.get_value(ftag).strip() if dpg.does_item_exist(ftag) else "5"
            nodes.append({
                "tid": tid,
                "tab": tab,
                "ntag": ntag,
                "label": label,
                "top_k": top_k,
                "field_tag": ftag,
            })
    return nodes


def _pick_checkpoint_node(prefer_filled: bool = True) -> dict | None:
    nodes = _iter_checkpoint_nodes()
    if not nodes:
        return None
    if prefer_filled:
        for node in nodes:
            if node["path"]:
                return node
    return nodes[0]


def _pick_output_node() -> dict | None:
    nodes = _iter_output_nodes()
    return nodes[0] if nodes else None


def _update_output_nodes_ui(status: str, summary: str, rows: list[tuple[str, tuple[int, int, int]]] | None = None) -> None:
    from ml_forge.graph.nodes import (
        inference_output_results_tag,
        inference_output_status_tag,
        inference_output_summary_tag,
    )

    for node in _iter_output_nodes():
        ntag = node["ntag"]
        status_tag = inference_output_status_tag(ntag)
        summary_tag = inference_output_summary_tag(ntag)
        results_tag = inference_output_results_tag(ntag)

        if dpg.does_item_exist(status_tag):
            dpg.set_value(status_tag, status)
            dpg.configure_item(
                status_tag,
                color=(220, 80, 80) if status.lower().startswith("error") else
                      (100, 180, 255) if "running" in status.lower() else
                      (80, 220, 120) if "complete" in status.lower() else
                      (140, 140, 140),
            )
        if dpg.does_item_exist(summary_tag):
            dpg.set_value(summary_tag, summary)
        if dpg.does_item_exist(results_tag):
            dpg.delete_item(results_tag, children_only=True)
            if rows:
                for text, color in rows:
                    dpg.add_text(text, color=color, parent=results_tag, wrap=220)
            else:
                dpg.add_text("Top-k results will appear here.", color=(120, 120, 120), parent=results_tag)


def _update_checkpoint_source_text(node: dict | None) -> None:
    if not dpg.does_item_exist("inf_ckpt_source"):
        return
    if node and node.get("path"):
        dpg.set_value(
            "inf_ckpt_source",
            f"Source: graph node {node['label']} ({node['tab']['name']})",
        )
    else:
        dpg.set_value("inf_ckpt_source", "Source: manual input")


def _sync_checkpoint_from_graph() -> dict | None:
    node = _pick_checkpoint_node(prefer_filled=True)
    if node and node.get("path"):
        _state["ckpt_path"] = _resolve_user_path(node["path"])
        _state["source"] = f"graph:{node['label']}"
        if dpg.does_item_exist("inf_ckpt_path"):
            dpg.set_value("inf_ckpt_path", _state["ckpt_path"])
    _update_checkpoint_source_text(node)
    return node


def _write_checkpoint_to_graph(path: str) -> None:
    node = _pick_checkpoint_node(prefer_filled=True) or _pick_checkpoint_node(prefer_filled=False)
    if not node:
        _state["source"] = "manual"
        _update_checkpoint_source_text(None)
        return
    if dpg.does_item_exist(node["field_tag"]):
        dpg.set_value(node["field_tag"], path)
    _state["source"] = f"graph:{node['label']}"
    _update_checkpoint_source_text({**node, "path": path})


def _resolve_checkpoint_path() -> str:
    if dpg.does_item_exist("inf_ckpt_path"):
        path = _resolve_user_path(dpg.get_value("inf_ckpt_path").strip())
        if path:
            _state["ckpt_path"] = path
            _write_checkpoint_to_graph(path)
            return path

    node = _sync_checkpoint_from_graph()
    return _resolve_user_path(node["path"]) if node and node.get("path") else ""


def _normalise_state_dict_keys(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        nk = key
        if nk.startswith("state_dict."):
            nk = nk[len("state_dict."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = value
    return cleaned


def _extract_state_dict(payload):
    if isinstance(payload, dict):
        if payload and all(hasattr(v, "shape") for v in payload.values()):
            return _normalise_state_dict_keys(payload)
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return _normalise_state_dict_keys(candidate)
    raise ValueError("Checkpoint does not contain a supported state_dict payload.")


def open_inference_window() -> None:
    tag = "inference_window"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)

    _ensure_texture()
    graph_node = _sync_checkpoint_from_graph()
    output_node = _pick_output_node()

    vw = dpg.get_viewport_client_width()
    vh = dpg.get_viewport_client_height()
    ww, wh = 520, 500
    px = (vw - ww) // 2
    py = (vh - wh) // 2

    with dpg.window(label="Inference", tag=tag, width=ww, height=wh,
                    pos=(px, py), no_collapse=True, modal=False):

        dpg.add_text("Runs the connected Inference graph and previews predictions.",
                     color=(160, 160, 160))
        dpg.add_spacer(height=8)

        dpg.add_text("Checkpoint (.pth / .pt / .ckpt)", color=(200, 200, 200))
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                tag="inf_ckpt_path",
                width=350,
                hint="path/to/checkpoint.(pth|pt|ckpt)",
                default_value=_state["ckpt_path"],
            )
            dpg.add_button(label="Browse", callback=_browse_checkpoint)
        dpg.add_text(
            f"Source: graph node {graph_node['label']} ({graph_node['tab']['name']})"
            if graph_node and graph_node.get("path") else
            "Source: manual input",
            tag="inf_ckpt_source",
            color=(120, 160, 200),
        )

        dpg.add_spacer(height=8)

        with dpg.group(horizontal=True):
            dpg.add_text("Top-k:", color=(180, 180, 180))
            dpg.add_slider_int(
                tag="inf_topk",
                default_value=max(1, _safe_int(output_node["top_k"], 5)) if output_node else 5,
                min_value=1,
                max_value=10,
                width=120,
            )

        dpg.add_spacer(height=8)

        with dpg.group(horizontal=True):
            dpg.add_button(label="New Sample", width=140, callback=_new_sample)
            with dpg.theme() as blue_th:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 100, 160, 220))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 130, 200, 255))
            dpg.bind_item_theme(dpg.last_item(), blue_th)

            dpg.add_spacer(width=8)

            dpg.add_button(label="Run Inference", width=140, callback=_run_on_current_sample)
            _apply_green_theme(dpg.last_item())

        dpg.add_spacer(height=4)
        dpg.add_text("", tag="inf_status", color=(140, 140, 140))
        dpg.add_separator()
        dpg.add_spacer(height=6)

        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Sample", color=(140, 140, 140))
                dpg.add_image(_TEX_TAG, width=_PREV_W, height=_PREV_H)
                dpg.add_text("", tag="inf_true_label", color=(120, 180, 120))

            dpg.add_spacer(width=16)

            with dpg.group():
                dpg.add_text("Predictions", color=(140, 140, 140))
                with dpg.child_window(tag="inf_results", width=220,
                                      height=_PREV_H, border=False):
                    dpg.add_text("Press 'New Sample' or RUN -> Inference to start.",
                                 color=(100, 100, 100))


def _apply_green_theme(item) -> None:
    with dpg.theme() as th:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 120, 60, 220))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 160, 80, 255))
    dpg.bind_item_theme(item, th)


def _browse_checkpoint() -> None:
    def _cb(sender, app_data):
        path = app_data.get("file_path_name", "")
        if path and dpg.does_item_exist("inf_ckpt_path"):
            resolved = _resolve_user_path(path)
            dpg.set_value("inf_ckpt_path", resolved)
            _state["ckpt_path"] = resolved
            _write_checkpoint_to_graph(resolved)

    tag = "inf_ckpt_dialog"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    with dpg.file_dialog(label="Select checkpoint", tag=tag,
                         width=600, height=400, callback=_cb, modal=True):
        dpg.add_file_extension(".pth")
        dpg.add_file_extension(".pt")
        dpg.add_file_extension(".ckpt")


def _get_inference_tab():
    from ml_forge.engine.graph import get_tab_by_role

    return get_tab_by_role("inference")


def _ancestors_of_target(tab: dict, ordered, target_node) -> list:
    if target_node is None:
        return []
    targets = {target_node.ntag}
    found = set()
    changed = True
    while changed:
        changed = False
        for _, (a1, a2) in tab["links"].items():
            if isinstance(a1, int):
                a1 = dpg.get_item_alias(a1) or str(a1)
            if isinstance(a2, int):
                a2 = dpg.get_item_alias(a2) or str(a2)
            dp = a2.split("_")
            sp = a1.split("_")
            if len(dp) >= 3 and len(sp) >= 3:
                dst = f"node_{dp[1]}_{dp[2]}"
                src = f"node_{sp[1]}_{sp[2]}"
                if dst in targets and src not in found:
                    found.add(src)
                    targets.add(src)
                    changed = True
    return [n for n in ordered if n.ntag in found]


def _build_transform(chain_nodes):
    from torchvision import transforms

    tlist = []
    for n in chain_nodes:
        label = n.block_label
        if label not in _INF_AUG_LABELS:
            continue
        if label == "Inf ToTensor":
            tlist.append(transforms.ToTensor())
        elif label == "Inf Resize":
            tlist.append(transforms.Resize(int(n.params.get("size", "224") or "224")))
        elif label == "Inf CenterCrop":
            tlist.append(transforms.CenterCrop(int(n.params.get("size", "224") or "224")))
        elif label == "Inf Normalize":
            mean = n.params.get("mean", "0.5").strip() or "0.5"
            std = n.params.get("std", "0.5").strip() or "0.5"
            try:
                mean = eval(mean) if "[" in mean or "(" in mean else [float(mean)] * 3
                std = eval(std) if "[" in std or "(" in std else [float(std)] * 3
            except Exception:
                mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            tlist.append(transforms.Normalize(mean=mean, std=std))
        elif label == "Inf Grayscale":
            tlist.append(transforms.Grayscale(int(n.params.get("num_output_channels", "1") or "1")))

    if not any(isinstance(t, transforms.ToTensor) for t in tlist):
        tlist.insert(0, transforms.ToTensor())
    return transforms.Compose(tlist)


def _load_dataset_from_tab(tab: dict | None):
    from torchvision import datasets
    from ml_forge.engine.graph import _INFERENCE_DATASET_BLOCKS, build_graph, topological_sort

    if tab is None:
        return None, "No Inference tab found."

    try:
        ordered = topological_sort(tab)
    except Exception as e:
        return None, f"Graph error: {e}"

    graph = build_graph(tab)
    nodes = list(graph.values())

    checkpoint_node = next((n for n in nodes if n.block_label in _CHECKPOINT_NODE_LABELS), None)
    chain = _ancestors_of_target(tab, ordered, checkpoint_node) or list(ordered)
    ds_node = next((n for n in chain if n.block_label in _INFERENCE_DATASET_BLOCKS), None)
    if ds_node is None:
        ds_node = next((n for n in ordered if n.block_label in _INFERENCE_DATASET_BLOCKS), None)
    if ds_node is None:
        return None, "No Dataset node found in Inference tab."

    transform = _build_transform(chain)
    label = ds_node.block_label
    root = _resolve_user_path(ds_node.params.get("root", "./data").strip() or "./data")
    download = ds_node.params.get("download", "True").strip()
    use_train = ds_node.params.get("train", "True").strip().lower() != "false"

    torch_datasets = {
        "Inf MNIST": datasets.MNIST,
        "Inf CIFAR10": datasets.CIFAR10,
        "Inf CIFAR100": datasets.CIFAR100,
        "Inf FashionMNIST": datasets.FashionMNIST,
    }

    try:
        if label in torch_datasets:
            ds = torch_datasets[label](
                root=root,
                train=use_train,
                download=(download.lower() != "false"),
                transform=transform,
            )
            return ds, None
        if label == "Inf ImageFolder":
            from torchvision.datasets import ImageFolder

            if not pathlib.Path(root).exists():
                return None, f"ImageFolder root not found: {root}"
            return ImageFolder(root=root, transform=transform), None
        return None, f"Dataset '{label}' not supported for inference sampling."
    except Exception as e:
        return None, str(e)


def _load_test_dataset():
    ds, err = _load_dataset_from_tab(_get_inference_tab())
    if ds is not None:
        return ds, None
    return None, err


def _get_random_sample():
    import torch
    from torchvision import transforms as T

    ds, err = _load_test_dataset()
    if ds is None:
        raise ValueError(err or "Could not load inference dataset.")

    idx = random.randint(0, len(ds) - 1)
    tensor, label = ds[idx]

    t_display = tensor.clone().detach()
    t_display = torch.clamp(t_display, 0.0, 1.0)
    pil_img = T.ToPILImage()(t_display)

    return pil_img, tensor, label


def _get_graph_topk() -> int:
    node = _pick_output_node()
    return max(1, _safe_int(node["top_k"], 5)) if node else 5


def _new_sample() -> None:
    _set_status("Loading sample...")
    _update_output_nodes_ui("Running...", "Loading a sample from the connected inference dataset...", None)
    _clear_results()
    try:
        pil_img, tensor, true_label = _get_random_sample()
        _state["last_sample"] = (pil_img, tensor, true_label)
        _pil_to_texture(pil_img)
        if dpg.does_item_exist("inf_true_label"):
            dpg.set_value("inf_true_label", f"True class: {true_label}")
        _set_status("Sample loaded. Press 'Run Inference' to classify.")
        _update_output_nodes_ui("Sample Ready", f"Loaded sample with true label {true_label}.", None)
    except Exception as e:
        _set_status(f"Error: {e}", error=True)
        _update_output_nodes_ui("Error", f"Sample load failed: {e}", None)
        log(f"Inference sample error: {e}", "error")


def _run_on_current_sample() -> None:
    if _state["last_sample"] is None:
        _new_sample()
        if _state["last_sample"] is None:
            return

    ckpt_path = _resolve_checkpoint_path()
    topk = dpg.get_value("inf_topk") if dpg.does_item_exist("inf_topk") else 5

    if not ckpt_path:
        _set_status("Select a checkpoint first, or fill a pth/pt/ckpt node.", error=True)
        return
    if not pathlib.Path(ckpt_path).exists():
        _set_status("Checkpoint file not found.", error=True)
        return

    try:
        import torch
    except ImportError:
        log("Inference: PyTorch not installed.", "error")
        return

    _set_status("Running model...")
    _update_output_nodes_ui("Running...", "Executing checkpoint on the current sample...", None)
    _clear_results()

    try:
        _, tensor, true_label = _state["last_sample"]

        from ml_forge.engine.run import _build_torch_model, _resolve_device

        device = _resolve_device("auto")
        model = _build_torch_model(device)
        model.eval()

        sd = _extract_state_dict(torch.load(ckpt_path, map_location=device))
        model.load_state_dict(sd)

        inp = tensor.unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(inp)
            probs = torch.softmax(logits, dim=1)[0]

        topk_vals, topk_idxs = torch.topk(probs, min(topk, probs.shape[0]))
        correct = int(true_label) if not isinstance(true_label, int) else true_label

        for rank, (idx, conf) in enumerate(zip(topk_idxs.tolist(), topk_vals.tolist()), 1):
            _show_result_row(rank, idx, conf, is_correct=(idx == correct))

        top1 = topk_idxs[0].item()
        top1_c = topk_vals[0].item()
        verdict = "CORRECT" if top1 == correct else "WRONG"
        _set_status(f"Top-1: class {top1} ({top1_c:.1%}) - {verdict}")
        _update_output_nodes_ui(
            "Complete",
            f"True class: {correct} | Top-1: {top1} ({top1_c:.1%}) | {verdict}",
            [
                (f"True class: {correct}", (120, 180, 120)),
                *[
                    (
                        f"#{rank}  class {idx:>4}   {conf*100:5.1f}%{'  v' if idx == correct else ''}",
                        (80, 220, 120) if idx == correct else (220, 180, 60) if conf > 0.3 else (180, 180, 180),
                    )
                    for rank, (idx, conf) in enumerate(zip(topk_idxs.tolist(), topk_vals.tolist()), 1)
                ],
            ],
        )
        log(f"Inference: pred={top1} conf={top1_c:.3f} true={correct} [{verdict}]", "success")

    except Exception as e:
        _clear_results()
        _show_result_text(f"Error: {e}", (220, 80, 80))
        _set_status(f"Model error: {e}", error=True)
        _update_output_nodes_ui("Error", f"Inference failed: {e}", None)
        log(f"Inference error: {e}", "error")


def run_inference_pipeline() -> None:
    from ml_forge.engine.graph import validate_inference_pipeline
    from ml_forge.ui.training import clear_highlights, highlight_issues

    result = validate_inference_pipeline()
    clear_highlights()

    if result.issues:
        highlight_issues(result.issues)
        log("-- Inference Validation --", "header")
        for issue in result.errors:
            log(f"ERROR: {issue.message}" + (f" [{issue.ntag}]" if issue.ntag else ""), "error")
        for issue in result.warnings:
            log(f"WARN:  {issue.message}" + (f" [{issue.ntag}]" if issue.ntag else ""), "warning")

    if not result.ok:
        _update_output_nodes_ui("Error", f"Inference validation failed with {len(result.errors)} error(s).", None)
        log(f"Cannot start inference - {len(result.errors)} error(s). Fix highlighted nodes.", "error")
        return

    open_inference_window()

    try:
        pil_img, tensor, true_label = _get_random_sample()
        _state["last_sample"] = (pil_img, tensor, true_label)
        _pil_to_texture(pil_img)
        if dpg.does_item_exist("inf_true_label"):
            dpg.set_value("inf_true_label", f"True class: {true_label}")
        if dpg.does_item_exist("inf_topk"):
            dpg.set_value("inf_topk", _get_graph_topk())
        _sync_checkpoint_from_graph()
        _set_status("Inference graph resolved. Running model...")
        _update_output_nodes_ui("Running...", "Inference graph validated. Preparing execution...", None)
        _run_on_current_sample()
    except Exception as e:
        _set_status(f"Error: {e}", error=True)
        _update_output_nodes_ui("Error", f"Inference pipeline failed: {e}", None)
        log(f"Inference pipeline error: {e}", "error")


def _sample_and_run() -> None:
    _new_sample()
    _run_on_current_sample()


def _clear_results() -> None:
    if dpg.does_item_exist("inf_results"):
        dpg.delete_item("inf_results", children_only=True)


def _show_result_text(msg: str, color=(160, 160, 160)) -> None:
    if dpg.does_item_exist("inf_results"):
        dpg.add_text(msg, color=color, parent="inf_results")


def _show_result_row(rank: int, class_idx: int, conf: float,
                     is_correct: bool = False) -> None:
    if not dpg.does_item_exist("inf_results"):
        return
    pct = conf * 100
    col = (80, 220, 120) if is_correct else (220, 180, 60) if conf > 0.3 else (180, 180, 180)
    marker = "  v" if is_correct else ""
    dpg.add_text(f"#{rank}  class {class_idx:>4}   {pct:5.1f}%{marker}",
                 color=col, parent="inf_results")
    dpg.add_progress_bar(default_value=conf, width=-1, overlay="", parent="inf_results")
    dpg.add_spacer(height=4, parent="inf_results")


def _set_status(msg: str, error: bool = False) -> None:
    if dpg.does_item_exist("inf_status"):
        col = (220, 80, 80) if error else (140, 140, 140)
        dpg.set_value("inf_status", msg)
        dpg.configure_item("inf_status", color=col)
