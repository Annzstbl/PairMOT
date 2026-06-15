# Data Preparation for Rotation Detection

It is recommended to symlink the dataset root to `$ai4rs/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

Datasets supported in ai4rs:

- [DOTA Dataset](dota/README.md) \[ [Homepage](https://captain-whu.github.io/DOTA/) \]
- [DIOR Dataset](dior/README.md) \[ [Homepage](https://gcheng-nwpu.github.io/#Datasets) \]
- [SSDD Dataset](ssdd/README.md)
- [HRSC Dataset](hrsc/README.md)
- [HRSID Dataset](hrsid/README.md)
- [SRSDD Dataset](srsdd/README.md)
- [RSDD Dataset](rsdd/README.md)
- [ICDAR2015 Dataset](icdar2015/README.md)
- [SARDet 100K Dataset](./sardet_100k/README.md)
- [RSAR Dataset](./rsar/README.md)
- [FAIR1M Dataset](./fair/README.md) \[ [Homepage](https://www.gaofen-challenge.com/benchmark) \]
- [ReCon1M Dataset](./recon1m/README.md) \[ [Homepage](https://recon1m-dataset.github.io/) \]
- [iSAID Dataset](./isaid/README.md) \[ [Homepage](https://captain-whu.github.io/iSAID/) \]
- [DroneVehicle Dataset](./dronevehicle/README.md)
- [NWPU VHR-10 Dataset](./nwpu/README.md)
- [HSMOT Dataset](./hsmot/README.md)

```
ai4rs
├── data
│   ├── split_ss_dota
│   │   ├── trainval
│   │   ├── test
│   ├── split_ms_dota
│   │   ├── trainval
│   │   ├── test
│   ├── split_ss_dota1.5
│   │   ├── trainval
│   │   ├── test
│   ├── DIOR
│   │   ├── Annotations
│   │   │   ├─ Oriented Bounding Boxes
│   │   │   ├─ Horizontal Bounding Boxes
│   │   ├── ImageSets
│   │   │   ├─ Main
│   │   │   │  ├─ train.txt
│   │   │   │  ├─ val.txt
│   │   │   │  ├─ test.txt
│   │   ├── JPEGImages-test
│   │   ├── JPEGImages-trainval
│   ├── icdar2015
│   │   ├── ic15_textdet_train_img
│   │   ├── ic15_textdet_train_gt
│   │   ├── ic15_textdet_test_img
│   │   ├── ic15_textdet_test_gt
│   ├── SARDet_100K
│   │   ├── Annotations
│   │   │   ├── test.json
│   │   │   ├── train.json
│   │   │   ├── val.json
│   │   ├── JPEGImages
│   │   │   ├── test
│   │   │   │   ├── 0000018.png
│   │   │   │   ├── xxxxxxx.png
│   │   │   ├── train
│   │   │   │   ├── xxxxxxx.png
│   │   │   │   ├── xxxxxxx.png
│   │   │   ├── val
│   │   │   │   ├── xxxxxxx.png
│   │   │   │   ├── xxxxxxx.png
│   ├── RSAR
│   │   ├── train
│   │   │   ├── annfiles
│   │   │   ├── images
│   │   ├── val
│   │   │   ├── annfiles
│   │   │   ├── images
│   │   ├── test
│   │   │   ├── annfiles
│   │   │   ├── images
│   ├── FAIR1M1.0
│   │   ├── train
│   │   │   ├── part1
│   │   │   │   ├── images
│   │   │   │   ├── labelXml
│   │   │   ├── part2
│   │   │   │   ├── images-1
│   │   │   │   ├── images-2
│   │   │   │   ├── labelXml
│   │   ├── test
│   │   │   ├── images
│   ├── split_ss_fair1m1.0
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── annfiles
│   ├── split_isaid
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── instance_only_filtered_train.json
│   │   ├── val
│   │   │   ├── images
│   │   │   ├── instance_only_filtered_val.json
│   │   ├── test
│   │   │   ├── images
│   ├── DroneVehicle
│   │   ├── train
│   │   │   │   ├── trainimg    # 17990 jpg
│   │   │   │   ├── trainimgr   # 17990 jpg
│   │   │   │   ├── trainlabel  # 17990 xml
│   │   │   │   ├── trainlabelr # 17990 xml
│   │   ├── val
│   │   │   │   ├── valimg      # 1469 jpg
│   │   │   │   ├── valimgr     # 1469 jpg
│   │   │   │   ├── vallabel    # 1469 xml
│   │   │   │   ├── vallabelr   # 1469 xml
│   │   ├── test
│   │   │   │   ├── testimg     # 8980 jpg
│   │   │   │   ├── testimgr    # 8980 jpg
│   │   │   │   ├── testlabel   # 8980 xml
│   │   │   │   ├── testlabelr  # 8980 xml
```
