import numpy as np

EPS = 1e-12

# Note on Objectives vs Drift Types:
# - Objectives (G/P) are CLIENT PROPERTIES assigned at initialization
#   They represent the client's strategic goal (generalization vs personalization)
# - Drift Types (real/label/feature/none) are ENVIRONMENTAL EVENTS detected at runtime
#   They represent what happened to the client's data distribution
# - The merge policy combines BOTH to decide routing strategy
DEFAULT_THRESHOLDS = {
    "prototype_shift": 0.1,
    "feature_shift": 0.3,
    "label_shift": 0.3,
    "accuracy_drop": 0.1,
}
DRIFT_TYPES = ("none", "label", "feature", "real")


def normalize_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return values / (total + EPS)


def label_distribution_shift(prev_dist, curr_dist, metric="l1"):
    prev = normalize_distribution(prev_dist)
    curr = normalize_distribution(curr_dist)

    if metric == "l1":
        return np.abs(prev - curr).sum()
    if metric == "l2":
        return np.linalg.norm(prev - curr)
    if metric == "js":
        m = 0.5 * (prev + curr)
        return 0.5 * (kl_divergence(prev, m) + kl_divergence(curr, m))
    raise ValueError(f"Unsupported metric '{metric}' for label distribution shift.")


def kl_divergence(p, q):
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    mask = (p > 0) & (q > 0)
    if not mask.any():
        return 0.0
    return float(np.sum(p[mask] * np.log((p[mask] + EPS) / (q[mask] + EPS))))


def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    if denom <= EPS:
        return 0.0
    cosine = np.dot(a, b) / denom
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def prototype_shift(prev_prototype, curr_prototype, metric="cosine"):
    if metric == "cosine":
        return cosine_distance(prev_prototype, curr_prototype)
    if metric == "l2":
        return float(np.linalg.norm(np.asarray(prev_prototype) - np.asarray(curr_prototype)))
    raise ValueError(f"Unsupported metric '{metric}' for prototype shift.")


def predict_labels(logits):
    logits = np.asarray(logits)
    if logits.ndim == 1:
        logits = logits[:, None]
    return np.argmax(logits, axis=1)


def accuracy_on_labels(logits, labels, label_subset=None):
    labels = np.asarray(labels)
    predictions = predict_labels(logits)

    if label_subset is not None:
        mask = np.isin(labels, list(label_subset))
        if not mask.any():
            return 0.0
        labels = labels[mask]
        predictions = predictions[mask]

    if labels.size == 0:
        return 0.0

    correct = (predictions == labels).sum()
    return float(correct) / float(labels.size)


def shared_label_accuracy_drop(last_logits, last_labels, current_logits, current_labels):
    last_labels = np.asarray(last_labels)
    current_labels = np.asarray(current_labels)
    shared = np.intersect1d(last_labels, current_labels)

    if shared.size == 0:
        # No shared label -> cannot compute drop reliably
        return 0.0, 0.0, 0.0

    last_acc = accuracy_on_labels(last_logits, last_labels, label_subset=shared)
    current_acc = accuracy_on_labels(current_logits, current_labels, label_subset=shared)
    return float(last_acc - current_acc), last_acc, current_acc


def diagnose_drift_type(metrics, thresholds=None, cluster_changed=False):
    thresholds = thresholds or DEFAULT_THRESHOLDS
    proto_shift = metrics.get("prototype_shift", 0.0)
    feature_shift_value = metrics.get("feature_shift", proto_shift)
    label_shift_value = metrics.get("label_shift", 0.0)
    accuracy_drop_value = abs(metrics.get("accuracy_drop", 0.0))

    if (
        proto_shift < thresholds["prototype_shift"]
        and feature_shift_value < thresholds["feature_shift"]
        and label_shift_value < thresholds["label_shift"]
        and accuracy_drop_value < thresholds["accuracy_drop"]
    ):
        return "none"

    if (
        label_shift_value >= thresholds["label_shift"]
        and feature_shift_value < thresholds["feature_shift"]
        and accuracy_drop_value < thresholds["accuracy_drop"]
    ):
        return "label"

    if (
        feature_shift_value >= thresholds["feature_shift"]
        and accuracy_drop_value < thresholds["accuracy_drop"]
    ):
        return "feature"

    if accuracy_drop_value >= thresholds["accuracy_drop"] or cluster_changed:
        return "real"

    # fallback
    return "real" if cluster_changed else "none"


def assign_objectives(clients, assignment_str, drift_types=None):
    """
    Assign objectives to clients. Mainly for static assignment (client goals).
    
    NOTE: For drift-based routing, you don't need this! Just use --merge_policy directly.
    The merge_policy can split clients 50/50 by index without needing objectives.
    
    Use this only for static client goals (hospital vs retailer scenarios).
    
    Args:
        clients: List of client objects
        assignment_str: Assignment specification string:
            - "all:G" or "all:P" - all clients get G or P
            - "first_half:G,second_half:P" - first half get G, second half get P
            - "drift_based:..." - DEPRECATED: Use --merge_policy directly instead!
        drift_types: Optional dict mapping client index -> drift type (for drift_based, but deprecated)
    
    Examples:
        # Static assignment (at initialization)
        assign_objectives(clients, "all:G")
        assign_objectives(clients, "first_half:G,second_half:P")
        
        # For drift-based routing: DON'T use assign_objectives!
        # Just use --merge_policy "label:halfP,halfG" directly
    """
    n_clients = len(clients)
    if n_clients == 0:
        return
    
    # Check if drift-based assignment
    if assignment_str.startswith("drift_based:"):
        if drift_types is None:
            raise ValueError("drift_based assignment requires drift_types parameter")
        
        # Parse drift-based mapping
        mapping_str = assignment_str[len("drift_based:"):]
        drift_to_strategy = {}
        
        for entry in mapping_str.split(','):
            entry = entry.strip()
            if ':' not in entry:
                continue
            # Handle nested colons: "label:halfP,halfG" or "real:allP"
            parts = entry.split(':', 1)
            if len(parts) != 2:
                continue
            drift_type = parts[0].strip()
            strategy = parts[1].strip()
            
            # Validate strategy
            valid_strategies = ("allG", "allP", "halfP,halfG", "halfG,halfP")
            if strategy not in valid_strategies:
                raise ValueError(
                    f"Invalid strategy '{strategy}' for drift type '{drift_type}'. "
                    f"Must be one of: {valid_strategies}"
                )
            drift_to_strategy[drift_type] = strategy
        
        # Group clients by drift type
        clients_by_drift = {}
        for i, client in enumerate(clients):
            drift_type = drift_types.get(i, "none")
            if drift_type not in clients_by_drift:
                clients_by_drift[drift_type] = []
            clients_by_drift[drift_type].append((i, client))
        
        # Assign objectives for each drift type
        for drift_type, client_list in clients_by_drift.items():
            strategy = drift_to_strategy.get(drift_type, "allG")  # Default to allG
            
            if strategy == "allG":
                # All clients with this drift type get G
                for _, client in client_list:
                    client.objective = "G"
            
            elif strategy == "allP":
                # All clients with this drift type get P
                for _, client in client_list:
                    client.objective = "P"
            
            elif strategy in ("halfP,halfG", "halfG,halfP"):
                # Half get P, half get G (deterministic split by index)
                n_clients_drift = len(client_list)
                mid_point = n_clients_drift // 2
                
                # Sort by client index for deterministic split
                client_list_sorted = sorted(client_list, key=lambda x: x[0])
                
                for idx, (_, client) in enumerate(client_list_sorted):
                    if idx < mid_point:
                        # First half get P
                        client.objective = "P"
                    else:
                        # Second half get G
                        client.objective = "G"
        
        return
    
    # Static assignment (original logic)
    if assignment_str.startswith("all:"):
        # All clients get same objective
        objective = assignment_str.split(":")[1].strip().upper()
        if objective not in ("G", "P"):
            raise ValueError(f"Invalid objective '{objective}'. Must be 'G' or 'P'.")
        for client in clients:
            client.objective = objective
    
    elif "first_half" in assignment_str and "second_half" in assignment_str:
        # Split by client ID (deterministic)
        parts = assignment_str.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid split format: '{assignment_str}'. Expected 'first_half:X,second_half:Y'")
        
        first_obj = None
        second_obj = None
        
        for part in parts:
            part = part.strip()
            if part.startswith("first_half:"):
                first_obj = part.split(":")[1].strip().upper()
            elif part.startswith("second_half:"):
                second_obj = part.split(":")[1].strip().upper()
        
        if first_obj is None or second_obj is None:
            raise ValueError(f"Invalid split format: '{assignment_str}'. Must specify both first_half and second_half")
        
        if first_obj not in ("G", "P") or second_obj not in ("G", "P"):
            raise ValueError(f"Objectives must be 'G' or 'P', got '{first_obj}' and '{second_obj}'")
        
        mid_point = n_clients // 2
        for i, client in enumerate(clients):
            if i < mid_point:
                client.objective = first_obj
            else:
                client.objective = second_obj
    
    else:
        raise ValueError(
            f"Invalid assignment string: '{assignment_str}'. "
            "Expected one of: 'all:G', 'all:P', 'first_half:G,second_half:P', "
            "or 'drift_based:label:G,real:P,feature:P,none:G'"
        )


def parse_merge_policy(policy_str):
    """
    Parse merge policy string into a dictionary.

    Accepts entries separated by comma or semicolon. Values may themselves contain
    a comma (e.g., "halfP,halfG"). Also accepts intuitive aliases:
      - reset  -> allP (no merge)
      - merge  -> allG (merge)
      - split  -> halfP,halfG (50/50 split or objective-aware split)

    Args:
        policy_str: String like "real:allP,label:halfP,halfG,feature:allG,none:halfP,halfG"
                    or using aliases: "real:reset;label:split;feature:merge;none:split"

    Returns:
        Dict mapping drift_type -> strategy: {"real": "allP", "label": "halfP,halfG", ...}
    """
    import re

    def _normalize(val: str) -> str:
        v = val.strip().lower()
        if v == 'reset':
            return 'allP'
        if v == 'merge':
            return 'allG'
        if v == 'split':
            return 'halfP,halfG'
        return val.strip()

    policy = {}
    if not policy_str:
        return policy

    # Support both comma and semicolon as separators between entries.
    s = policy_str.strip()

    # Regex: key:value where value may include one optional ",..." segment; stop before next key: or end
    pattern = re.compile(r"\b(real|label|feature|none|shift|clean)\s*:\s*([^,;]+(?:,[^,;]+)?)", re.IGNORECASE)
    for m in pattern.finditer(s):
        key = m.group(1).lower()
        val = _normalize(m.group(2))
        policy[key] = val

    return policy


def apply_drift_routing(clients, routing_policy_str=None, diagnosis_mode='multiclass', use_objectives=False):
    """
    Apply routing directly based on drift types. Simplified and more intuitive!
    
    Note: Reclustering is automatic (server-side) when prototypes shift clusters.
    This function only controls merging (client-side continual learning).
    
    By convention:
      - concept_shift_flag = 1: No merge (train on current data only)
      - concept_shift_flag = 0: Merge (train on old+new data for continual learning)
    
    Args:
        clients: List of client objects
        routing_policy_str: Policy string. Format:
            - "real:allP,label:halfP,halfG,feature:allG,none:allP"
            - Strategies: "allP" (all no merge), "allG" (all merge), "halfP,halfG" (split 50/50)
        diagnosis_mode: 'binary' or 'multiclass'
        use_objectives: If True, use client.objective for "halfP,halfG" routing.
                       If False, split 50/50 by client index (no objectives needed).
    
    Routing strategies:
      - "allP": All clients no merge (flag = 1)
      - "allG": All clients merge (flag = 0)
      - "halfP,halfG": 
          - If use_objectives=True: P-clients no merge, G-clients merge
          - If use_objectives=False: Split 50/50 by index, first half no merge, second half merge
    
    Example (use_objectives=False - simpler!):
        Label drift clients: strategy="halfP,halfG"
        → Split by index: first half no merge, second half merge
        → No objectives needed!
    
    Example (use_objectives=True - for static client goals):
        Client 0: objective="G" (hospital)
        Client 30: objective="P" (retailer)
        Label drift: strategy="halfP,halfG"
        → Client 0: merge (G-client)
        → Client 30: no merge (P-client)
        → Routes based on client goals
    """
    # Parse routing policy if provided
    if routing_policy_str:
        policy = parse_merge_policy(routing_policy_str)
    else:
        # Default policy based on diagnosis mode
        if diagnosis_mode == 'binary':
            policy = {
                "real": "allP",
                "none": "halfP,halfG"
            }
            policy["shift"] = policy["real"]
            policy["clean"] = policy["none"]
        else:
            policy = {
                "real": "allP",
                "label": "halfP,halfG",
                "feature": "halfP,halfG",
                "none": "halfP,halfG"
            }
    
    # Group clients by drift type for efficient processing
    clients_by_drift = {}
    for i, client in enumerate(clients):
        drift = getattr(client, "drift_type", "none")
        
        # Handle binary mode: map 'shift'/'clean' to 'real'/'none' if needed
        if diagnosis_mode == 'binary' and drift in ("shift", "clean"):
            drift = "real" if drift == "shift" else "none"
        
        if drift not in clients_by_drift:
            clients_by_drift[drift] = []
        clients_by_drift[drift].append((i, client))
    
    # Apply routing for each drift type
    for drift_type, client_list in clients_by_drift.items():
        strategy = policy.get(drift_type, "halfP,halfG")  # Default to split
        strategy_lc = (strategy or "").lower()
        
        if strategy_lc == "follow":
            # Always route by objectives (P=no-merge, G=merge)
            for _, client in client_list:
                objective = getattr(client, "objective", "G")
                client.concept_shift_flag = 1 if objective == "P" else 0
        elif strategy == "allP":
            # All clients: no merge (fast adaptation)
            for _, client in client_list:
                client.concept_shift_flag = 1
        elif strategy == "allG":
            # All clients: merge (continual learning)
            for _, client in client_list:
                client.concept_shift_flag = 0
        elif strategy in ("halfP,halfG", "halfG,halfP"):
            # Split strategy
            if use_objectives:
                # Route based on client objectives (for static client goals)
                for _, client in client_list:
                    objective = getattr(client, "objective", "G")
                    if objective == "P":
                        client.concept_shift_flag = 1  # No merge
                    else:  # objective == "G"
                        client.concept_shift_flag = 0  # Merge
            else:
                # Split 50/50 by client index (simpler, no objectives needed!)
                n_clients = len(client_list)
                mid_point = n_clients // 2
                client_list_sorted = sorted(client_list, key=lambda x: x[0])
                
                for idx, (_, client) in enumerate(client_list_sorted):
                    if idx < mid_point:
                        client.concept_shift_flag = 1  # First half: no merge
                    else:
                        client.concept_shift_flag = 0  # Second half: merge
        else:
            # Unknown strategy, default to split by index
            n_clients = len(client_list)
            mid_point = n_clients // 2
            client_list_sorted = sorted(client_list, key=lambda x: x[0])
            for idx, (_, client) in enumerate(client_list_sorted):
                client.concept_shift_flag = 1 if idx < mid_point else 0

# Backward compatibility alias
def apply_objective_policy(clients, merge_policy_str=None, diagnosis_mode='multiclass'):
    """Backward compatibility wrapper. Use apply_drift_routing instead."""
    # Check if any clients have objectives set (for static assignment)
    has_objectives = any(hasattr(c, "objective") and getattr(c, "objective", None) in ("G", "P") for c in clients)
    use_objectives = has_objectives
    return apply_drift_routing(clients, merge_policy_str, diagnosis_mode, use_objectives=use_objectives)
