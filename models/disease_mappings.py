# disease_mappings.py - Single source of truth for disease names and mappings

# Internal model class names (used by model outputs)
CLASS_NAMES = [
    'Acne', 'Actinic_Keratosis', 'Benign_tumors', 'Bullous', 
    'Candidiasis', 'DrugEruption', 'Eczema', 'Infestations_Bites', 
    'Lichen', 'Lupus', 'Moles', 'Psoriasis', 'Rosacea', 
    'Seborrh_Keratoses', 'SkinCancer', 'Sun_Sunlight_Damage', 
    'Tinea', 'Unknown_Normal', 'Vascular_Tumors', 'Vasculitis', 
    'Vitiligo', 'Warts'
]

# Mapping from internal model names to display names
name_mapping = {
    'Acne': 'Acne',
    'Actinic_Keratosis': 'Actinic Keratosis',
    'Benign_tumors': 'Benign Tumors',
    'Bullous': 'Bullous Disorders',
    'Candidiasis': 'Candidiasis',
    'DrugEruption': 'Drug Eruption',
    'Eczema': 'Eczema',
    'Infestations_Bites': 'Infestations/Bites',
    'Lichen': 'Lichen Planus',
    'Lupus': 'Lupus',
    'Moles': 'Moles',
    'Psoriasis': 'Psoriasis',
    'Rosacea': 'Rosacea',
    'Seborrh_Keratoses': 'Seborrheic Keratoses',
    'SkinCancer': 'Skin Cancer',
    'Sun_Sunlight_Damage': 'Sun/Sunlight Damage',
    'Tinea': 'Tinea',
    'Unknown_Normal': 'Unknown/Normal Variants',
    'Vascular_Tumors': 'Vascular Tumors',
    'Vasculitis': 'Vasculitis',
    'Vitiligo': 'Vitiligo',
    'Warts': 'Warts'
}

def get_display_name(model_name):
    """Convert model's internal name to display name"""
    return name_mapping.get(model_name, model_name.replace('_', ' '))

def get_model_name(display_name):
    """Convert display name back to model's internal name"""
    reverse_mapping = {v: k for k, v in name_mapping.items()}
    return reverse_mapping.get(display_name, display_name.replace(' ', '_'))
