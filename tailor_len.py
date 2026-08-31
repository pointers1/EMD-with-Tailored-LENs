import re
import copy
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch_explain.logic.nn import entropy


# ============================================================
# HELPERS
# ============================================================
def get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def to_numpy_1d(y):
    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy()
    if isinstance(y, pd.Series):
        return y.values
    return np.asarray(y)


# ============================================================
# FAST LOGICAL QUERY EVALUATION
# ============================================================
def _clean_formula(query):
    query = str(query)
    query = query.replace("∧", "&").replace("∨", "|").replace("¬", "~")
    query = query.replace("\n", " ")
    query = re.sub(r"\s+", " ", query).strip()
    return query


def _literal_to_mask(literal, x):
    literal = literal.strip()
    literal = literal.replace("(", "").replace(")", "").strip()

    if literal == "":
        return np.ones(len(x), dtype=bool)

    negated = literal.startswith("~")

    if negated:
        feature = literal[1:].strip()
    else:
        feature = literal.strip()

    if feature not in x.columns:
        raise ValueError(f"Feature '{feature}' from explanation not found in dataframe columns.")

    values = x[feature].values

    if negated:
        return values == 0
    else:
        return values == 1


def logical_expression_to_mask(query, x):
    query = _clean_formula(query)

    if query == "":
        return np.zeros(len(x), dtype=bool)

    or_terms = re.split(r"\s*\|\s*", query)
    final_mask = np.zeros(len(x), dtype=bool)

    for term in or_terms:
        term = term.strip()

        if term == "":
            continue

        and_literals = re.split(r"\s*&\s*", term)
        term_mask = np.ones(len(x), dtype=bool)

        for literal in and_literals:
            term_mask &= _literal_to_mask(literal, x)

        final_mask |= term_mask

    return final_mask


def query_test(logical_expression, df):
    mask = logical_expression_to_mask(logical_expression, df)
    return df.loc[mask]


def predict_with_explanation(query, x):
    mask = logical_expression_to_mask(query, x)
    return mask.astype(int)


# ============================================================
# EVALUATION
# ============================================================
def evaluate_exp(query, x, y):
    y_true = to_numpy_1d(y).astype(int)
    y_pred = predict_with_explanation(query, x)

    accuracy_custom = accuracy_score(y_true, y_pred)
    precision_custom = precision_score(y_true, y_pred, zero_division=0)
    f1_custom = f1_score(y_true, y_pred, zero_division=0)

    return precision_custom, accuracy_custom, f1_custom


def evaluate_explanation(query, x, y):
    y_true = to_numpy_1d(y).astype(int)
    y_pred = predict_with_explanation(query, x)

    accuracy_custom = accuracy_score(y_true, y_pred)
    precision_custom = precision_score(y_true, y_pred, zero_division=0)
    recall_custom = recall_score(y_true, y_pred, zero_division=0)
    f1_custom = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix_custom = confusion_matrix(y_true, y_pred)

    tn_custom, fp_custom, fn_custom, tp_custom = conf_matrix_custom.ravel()

    tpr_custom = tp_custom / (tp_custom + fn_custom) if (tp_custom + fn_custom) > 0 else 0
    fpr_custom = fp_custom / (fp_custom + tn_custom) if (fp_custom + tn_custom) > 0 else 0

    print("Explanation performance:")
    print("Accuracy:", accuracy_custom)
    print("Precision:", precision_custom)
    print("Recall:", recall_custom)
    print("F1 Score:", f1_custom)
    print("True Positive Rate (TPR):", tpr_custom)
    print("False Positive Rate (FPR):", fpr_custom)


# ============================================================
# MODEL PREDICTION
# ============================================================
def get_prediction(model, X):
    model.eval()
    device = get_model_device(model)

    if isinstance(X, pd.DataFrame):
        X_tensor = torch.tensor(X.values, dtype=torch.float32, device=device)
    elif isinstance(X, torch.Tensor):
        X_tensor = X.detach().to(device).float()
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        y_pred_test = model(X_tensor).squeeze(-1)
        probs = torch.sigmoid(y_pred_test)

        if probs.ndim == 2 and probs.shape[1] == 2:
            predicted = (probs[:, 1] >= 0.5).long()
        else:
            predicted = (probs >= 0.5).long()

    return predicted.detach().cpu().numpy()


# ============================================================
# SIMPLIFICATION
# ============================================================
def simplify(conditions, X, y, threshold):
    valid_sub_conditions = []

    for sub_condition in conditions:
        sub_condition = str(sub_condition).strip()

        if sub_condition == "":
            continue

        precision, _, _ = evaluate_exp(sub_condition, X, y)

        if precision >= threshold:
            valid_sub_conditions.append(sub_condition)

    return " | ".join(valid_sub_conditions)


def simplify_formula(explanation, x, y):
    if explanation is None or str(explanation).strip() == "":
        return ""

    explanation = str(explanation).strip()
    _, base_accuracy, _ = evaluate_exp(explanation, x, y)

    terms = explanation.split(" | ")

    for term in terms:
        term = term.strip()

        if term == "":
            continue

        explanation_simplified = copy.deepcopy(explanation)

        if explanation_simplified == term:
            continue

        if explanation_simplified.endswith(term):
            explanation_simplified = explanation_simplified.replace(f" | {term}", "")
        else:
            explanation_simplified = explanation_simplified.replace(f"{term} | ", "")

        explanation_simplified = explanation_simplified.strip()

        if explanation_simplified:
            _, accuracy, _ = evaluate_exp(explanation_simplified, x, y)

            if accuracy >= base_accuracy:
                explanation = copy.deepcopy(explanation_simplified)
                base_accuracy = accuracy

    return explanation


# ============================================================
# MAIN EXPLANATION FUNCTION
# ============================================================
def get_explanation(model, X, y, concepts):
    """
    CPU-safe explanation extraction.

    Important:
    torch_explain.entropy.explain_classes internally calls .numpy(),
    so this part must run on CPU.
    """
    model.eval()

    original_device = get_model_device(model)

    model.to("cpu")
    model.eval()

    if isinstance(X, pd.DataFrame):
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
    elif isinstance(X, torch.Tensor):
        X_tensor = X.detach().cpu().float()
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)

    y_np = to_numpy_1d(y).astype(int)
    y_tensor = torch.tensor(y_np, dtype=torch.int64)

    concepts = list(concepts)

    explanations = entropy.explain_classes(
        model,
        X_tensor,
        y_tensor,
        c_threshold=0.5,
        y_threshold=0,
        class_names=["benign", "malware"],
        concept_names=concepts
    )[0]["1"]["explanation"]

    model.to(original_device)
    model.eval()

    expressions = re.split(r"\s*\|\s*", str(explanations))
    unique_expressions = sorted(
        list(set(expr.strip() for expr in expressions if expr.strip() != ""))
    )

    predicted_labels = get_prediction(model, X)
    precision = precision_score(y_np, predicted_labels, zero_division=0)

    initial_threshold = round(precision, 3)
    current_threshold = initial_threshold

    best_solution = simplify(
        unique_expressions,
        X,
        y_np,
        round(current_threshold, 3)
    )

    if best_solution == "":
        return ""

    _, best_accuracy, _ = evaluate_exp(best_solution, X, y_np)

    # Search downward
    while True:
        new_threshold = current_threshold - 0.005

        if new_threshold < initial_threshold - 0.2:
            break

        new_solution = simplify(
            unique_expressions,
            X,
            y_np,
            round(new_threshold, 3)
        )

        if new_solution == "":
            break

        _, new_accuracy, _ = evaluate_exp(new_solution, X, y_np)

        if new_accuracy >= best_accuracy:
            best_solution = new_solution
            best_accuracy = new_accuracy
            current_threshold = new_threshold
        else:
            break

    # Search upward
    current_threshold = initial_threshold + 0.005

    while True:
        if current_threshold > initial_threshold + 0.2:
            break

        new_solution = simplify(
            unique_expressions,
            X,
            y_np,
            round(current_threshold, 3)
        )

        if new_solution == "":
            break

        _, new_accuracy, _ = evaluate_exp(new_solution, X, y_np)

        if new_accuracy >= best_accuracy:
            best_solution = new_solution
            best_accuracy = new_accuracy
            current_threshold += 0.005
        else:
            break

    return simplify_formula(best_solution, X, y_np)


# ============================================================
# OTHER UTILITIES
# ============================================================
def replace_and_simplify_logic(input_string, X):
    def replace_feature(match):
        feature_num = int(match.group(2))
        return str(X.iloc[:, feature_num])

    pattern = re.compile(r"(feature(\d+))")
    replaced_string = pattern.sub(replace_feature, input_string)

    replaced_string = (
        replaced_string
        .replace("&", "∧")
        .replace("|", "∨")
        .replace("~", "¬")
    )

    expressions = re.split(r"\∨", replaced_string)
    expressions = [expr.strip() for expr in expressions]

    unique_expressions = list(set(expressions))
    simplified_logic_string = " ∨ ".join(unique_expressions)

    return simplified_logic_string


def parsify_explanation(input_string, X, y):
    def replace_feature(match):
        feature_num = int(match.group(2))
        return str(X.iloc[:, feature_num])

    pattern = re.compile(r"(feature(\d+))")
    replaced_string = pattern.sub(replace_feature, input_string)

    expressions = re.split(r"\|", replaced_string)
    expressions = [expr.strip() for expr in expressions]

    unique_expressions = list(set(expressions))
    simplified_logic_string = " | ".join(unique_expressions)

    return simplified_logic_string


def check_complexity(logical_condition):
    condition_without_parentheses = re.sub(r"\(|\)", "", logical_condition)

    conditions_split = re.split(
        r"\s*&\s*|\s*\|\s*",
        condition_without_parentheses
    )

    conditions_split = [condition for condition in conditions_split if condition]

    return f"Complexity: {len(conditions_split)}"