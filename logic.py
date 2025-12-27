LABEL_MAP = {
    0: "NEGATIVE",
    1: "POSITIVE"
}

CONFIDENCE_THRESHOLD = 0.75

def post_process(prediction: dict):
    label_id = prediction["label_id"]
    confidence = prediction["confidence"]

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "label": "UNCERTAIN",
            "confidence": confidence
        }

    return {
        "label": LABEL_MAP[label_id],
        "confidence": confidence
    }
