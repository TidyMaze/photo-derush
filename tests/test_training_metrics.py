import joblib
from PIL import Image

from src.training_core import train_keep_trash_model


class Repo:
    def __init__(self, mapping): self.mapping=mapping
    def get_state(self, f): return self.mapping.get(f,'')
    def get_label_source(self, f): return 'manual' if f in self.mapping else ''

def _make(path, color):
    Image.new('RGB',(40,30),color).save(path,'PNG')

def test_extended_metrics(tmp_path):
    # 10 images (5 keep,5 trash) for metrics & small CV
    mapping={}
    for i in range(5):
        k=f'k{i}.png'; t=f't{i}.png'
        _make(str(tmp_path/k),(250-i*5,10,10))
        _make(str(tmp_path/t),(10,250-i*5,10))
        mapping[k]='keep'; mapping[t]='trash'
    repo=Repo(mapping)
    model_path=tmp_path/'model.joblib'
    res=train_keep_trash_model(str(tmp_path), model_path=str(model_path), repo=repo, n_estimators=50, random_state=0)  # type: ignore[arg-type]
    assert res is not None
    assert res.precision is not None
    assert res.f1 is not None
    # roc_auc may be None for tiny dataset but should attempt
    data=joblib.load(model_path)
    assert 'precision' in data
    assert 'f1' in data
