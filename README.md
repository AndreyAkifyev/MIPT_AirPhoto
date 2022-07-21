# satellite-hackathon

Скачайте данные и положите в папку ```data```:

в ```configs/default_config.py``` измените переменную  ```cfg.data_dir``` на нужную, например ```./data/```

## Создать виртуальное окружение

```
python3 -m venv satellite-venv
source satellite-venv/bin/activate
bash create_env.sh
```

## Создать тренировочную подложку

```
source satellite-venv/bin/activate
python3 recreate_test.py
python3 merge_with_canvas.py
```

## Тренировка модели
~2000 эпох должно хватит для того чтобы модель сошлась к нужному значению

```
python3 train_model.py --config config_9
```

## Создать посылку
```
python3 inference.py --exp exp_1
```
