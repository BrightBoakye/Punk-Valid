import openai
import torch
import config
from dataset import to_device, get_device, get_image
from model import get_model

# Set up OpenAI API credentials
openai.api_key = " api-key "

# Function to call OpenAI API and get insights and recommendations
def get_recommendation(predicted_label):
    prompt = f"Explain {predicted_label} land type and the best practices for optimizing yield in {predicted_label} land type?"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.8,
    )
    recommendation = response.choices[0].text.strip() 
    return recommendation


def decode_target(target, text_labels=False):
    """Decode target labels into text (or not)"""
    if not text_labels:
        return target
    else:
        return config.IDX_CLASS_LABELS[target]


def predict_single(image_path):
    device = get_device()
    image = get_image(image_path, device)
    model = get_model(device)
    model.eval()
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    with torch.no_grad():
        preds = model(xb)
    _, prediction = torch.max(preds.cpu().detach(), dim=1)

    # Get insights and recommendations using OpenAI API
    predicted_label = decode_target(int(prediction), text_labels=True)
    recommendation = get_recommendation(predicted_label)

    return predicted_label, recommendation
