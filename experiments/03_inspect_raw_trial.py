import json
from pprint import pprint

with open("data/raw/clinical_trials_raw.json", "r", encoding="utf-8") as f:
    trials = json.load(f)

trial = trials[0]
protocol = trial["study"]["protocolSection"]

fields = {
    "identificationModule": protocol.get("identificationModule", {}),
    "statusModule": protocol.get("statusModule", {}),
    "descriptionModule": protocol.get("descriptionModule", {}),
    "conditionsModule": protocol.get("conditionsModule", {}),
    "designModule": protocol.get("designModule", {}),
    "armsInterventionsModule": protocol.get("armsInterventionsModule", {}),
    "eligibilityModule": protocol.get("eligibilityModule", {}),
}

for name, data in fields.items():
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    pprint(data)