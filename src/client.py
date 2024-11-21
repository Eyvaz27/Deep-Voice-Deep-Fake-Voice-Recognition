import requests
from analysis.scripts.data_load import sample_default, LOCAL_DATA_PATH

DATA_PATH = "/workspaces/Deep-Voice-Deep-Fake-Voice-Recognition/dataset/deep_voice.csv"

default_audio_track = sample_default(data_path=LOCAL_DATA_PATH)
default_audio_track = default_audio_track.to_dict()
url = "https://stunning-space-happiness-pp547j4rwqp3rr9-9696.app.github.dev/predict"

print(requests.post(url, json=default_audio_track).json())
print(requests.get(url).json())