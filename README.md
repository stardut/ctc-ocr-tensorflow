## mnist with rnn

### Requirements

 * python 3.5+
 * tensorflow (version >= 1.4)
 * numpy
 * opencv 3.2.0

### Data
<img src="test_data/0_05771840.jpg" width="224" height="28">、<img src="test_data/1_66454652.jpg" width="224" height="28">、<img src="test_data/2_13892770.jpg" width="224" height="28">

### Predict

```
python predict.py image_path
```

#### example:

```
python predict.py test_data/0_05771840.jpg
```

#### result
<img src="show.png" width="607" height="194">

### Train

```
python train.py
```

