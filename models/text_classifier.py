# text_classifier.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from disease_mappings import CLASS_NAMES, name_mapping, get_display_name


class SymptomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        # Join list of symptoms into a single string
        self.texts = df['Symptoms'].apply(
            lambda lst: ' '.join(lst) if isinstance(lst, list) else str(lst)
        ).tolist()
        # Encode labels
        self.labels = pd.Categorical(df['Disease']).codes
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc.input_ids.squeeze(0),
            'attention_mask': enc.attention_mask.squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_text_model(csv_path, checkpoint_path, batch_size=16, lr=2e-5, epochs=3):
    df = pd.read_csv(csv_path)
    num_labels = df['Disease'].nunique()

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = SymptomDataset(df, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_corrects = 0
        for batch in loader:
            optim.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            loss = outputs.loss
            loss.backward()
            optim.step()

            total_loss += loss.item() * batch['input_ids'].size(0)
            preds = outputs.logits.argmax(dim=1)
            total_corrects += (preds == batch['labels'].to(device)).sum().item()

        epoch_loss = total_loss / len(dataset)
        epoch_acc = total_corrects / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved text model to {checkpoint_path}")


# Dictionary mapping diseases to their symptoms (using display names from disease_mappings.py)
disease_to_symptoms = {
    name_mapping['Acne']: ["Pimples", "Blackheads", "Whiteheads", "Inflamed skin", "Oily skin", "Cystic acne"],
    name_mapping['Actinic_Keratosis']: ["Dry, scaly patches", "Red or brown spots", "Crusting", "Itchy spots", "Sun damage"],
    name_mapping['Benign_tumors']: ["Non-cancerous growths", "Soft lumps", "Flesh-colored bumps"],
    name_mapping['Bullous']: ["Large fluid-filled blisters", "Itchy blisters", "Painful blisters"],
    name_mapping['Candidiasis']: ["Red, sore skin", "Itchy rash", "White patches", "Swollen skin", "Flaky skin"],
    name_mapping['DrugEruption']: ["Rash", "Red spots", "Itching", "Hives", "Fever", "Swelling"],
    name_mapping['Eczema']: ["Itchy skin", "Redness", "Dry patches", "Swollen skin", "Cracked skin", "Blisters"],
    name_mapping['Infestations_Bites']: ["Itchy rash", "Visible bites", "Swollen red spots", "Wheals", "Lice"],
    name_mapping['Lichen']: ["Purple flat-topped bumps", "Itchy lesions", "White lines", "Inflammation"],
    name_mapping['Lupus']: ["Butterfly-shaped rash", "Red patches", "Scaly skin", "Sensitivity to sunlight", "Mouth sores"],
    name_mapping['Moles']: ["Dark spots", "Irregular border", "Itchy moles", "Raised moles", "Multicolored moles"],
    name_mapping['Psoriasis']: ["Scaly skin", "Itchy patches", "Red skin", "Thickened nails", "Dry cracked skin"],
    name_mapping['Rosacea']: ["Redness", "Visible blood vessels", "Pimple-like bumps", "Burning sensation", "Eye irritation"],
    name_mapping['Seborrh_Keratoses']: ["Waxy, raised lesions", "Brown, black, or light-colored growths", "Itchy spots"],
    name_mapping['SkinCancer']: ["Irregular borders", "Asymmetrical growth", "New growths", "Changes in existing moles", "Bleeding lesions"],
    name_mapping['Sun_Sunlight_Damage']: ["Sunburn", "Freckles", "Wrinkles", "Dark spots", "Rough patches"],    name_mapping['Tinea']: ["Red, ring-shaped rash", "Itchy rash", "Peeling skin", "Burning sensation", "Blisters"],
    name_mapping['Unknown_Normal']: ["No symptoms", "Minor blemishes", "Normal pigmentation"],
    name_mapping['Vascular_Tumors']: ["Red or purple spots", "Visible blood vessels", "Swelling", "Painful nodules"],
    name_mapping['Vasculitis']: ["Red or purple patches", "Swollen blood vessels", "Skin sores", "Rash", "Fever"],
    name_mapping['Vitiligo']: ["White patches", "Loss of pigmentation", "Irregular skin color", "Lightened skin"],
    name_mapping['Warts']: ["Raised growths", "Rough texture", "Itchy warts", "Hard lumps", "Bumps on skin"]
}

def get_unique_symptoms():
    """Extract unique symptoms from the provided mapping and return them in alphabetical order."""
    unique_symptoms = set()
    for symptoms in disease_to_symptoms.values():
        unique_symptoms.update(symptoms)

    # Convert to list and sort alphabetically
    return sorted(list(unique_symptoms))


if __name__ == "__main__":
    # Hardcoded paths
    csv_path = r"C:/Users/trafl/Desktop/Minor/Dataset/Text-based/skin_disease_symptoms_dataset.csv"
    ckpt_path = r"C:/Users/trafl/Desktop/Minor/Model/text_model.pth"

    train_text_model(csv_path, ckpt_path)
