import os
import requests
import pandas as pd
import re
from datetime import datetime

hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
my_token = f"Bearer {hf_token}"

headers = {
    "Authorization": my_token
}

def get_organization_models(org):
    stat_url = f"https://huggingface.co/api/models?author={org}"
    response = requests.get(stat_url, headers=headers)
    html = response.json()
    organization_models = pd.DataFrame([{"id": x['id']} for x in html])
    
    # Filter out unwanted models
    organization_models = organization_models[~organization_models['id'].str.contains("ModelCardReview|cp.|GGUF")]
    
    return organization_models

def get_organization_datasets(org):
    stat_url = f"https://huggingface.co/api/datasets?author={org}"
    response = requests.get(stat_url, headers=headers)
    html = response.json()
    organization_datasets = pd.DataFrame([{"id": x['id']} for x in html])
    
    # Filter out unwanted datasets
    organization_datasets = organization_datasets[~organization_datasets['id'].str.contains("ft")]
    
    return organization_datasets

def get_download_stats(url, type="model"):
    if type == "model":
        stat_url = f"https://huggingface.co/api/models/{url}?expand[]=downloads&expand[]=downloadsAllTime"
    elif type == "dataset":
        stat_url = f"https://huggingface.co/api/datasets/{url}?expand[]=downloads&expand[]=downloadsAllTime"
    
    response = requests.get(stat_url, headers=headers)
    html = response.json()
    
    downloads = int(re.sub(r",", "", str(html.get("downloads", 0))))
    downloads_all_time = int(re.sub(r",", "", str(html.get("downloadsAllTime", 0))))
    
    return {
        "model_url": url,
        "downloads": downloads,
        "downloadsAllTime": downloads_all_time
    }

if __name__ == "__main__":
    # Example usage
    org = "taide"
    taide_models = get_organization_models(org)
    taide_datasets = get_organization_datasets(org)

    # Fetch download stats for models and datasets
    df_model = pd.DataFrame([get_download_stats(x, "model") for x in taide_models['id']])
    df_dataset = pd.DataFrame([get_download_stats(x, "dataset") for x in taide_datasets['id']])

    # Filter out tokenizers from models
    df_model = df_model[~df_model['model_url'].str.contains("tokenizer")]

    # Extract organization and model/dataset name
    df_model['organization'] = df_model['model_url'].str.extract(r'(.*)(?=/)')
    df_dataset['organization'] = df_dataset['model_url'].str.extract(r'(.*)(?=/)')

    df_model['model_name'] = df_model['model_url'].str.extract(r'([^/]+$)')
    df_dataset['model_name'] = df_dataset['model_url'].str.extract(r'([^/]+$)')

    # Add current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    df_model['date'] = current_date
    df_dataset['date'] = current_date

    # Create directory and save the data to CSV files
    os.makedirs("example-data/models", exist_ok=True)
    os.makedirs("example-data/datasets", exist_ok=True)

    df_model.to_csv(f"example-data/models/{current_date}_hf.csv", index=False)
    df_dataset.to_csv(f"example-data/datasets/{current_date}_hf.csv", index=False)
