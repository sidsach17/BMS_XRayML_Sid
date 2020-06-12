from azureml.core import Dataset

xrayimage_dataset = Dataset.get_by_name(ws, name='xray_image_ds')
traindata_dataset = Dataset.get_by_name(ws, name='train_data_ds')
validdata_dataset = Dataset.get_by_name(ws, name='valid_data_ds')
testdata_dataset = Dataset.get_by_name(ws, name='test_data_ds')
traintarget_dataset = Dataset.get_by_name(ws, name='train_target_ds')
validtarget_dataset = Dataset.get_by_name(ws, name='valid_target_ds')
testtarget_dataset = Dataset.get_by_name(ws, name='test_target_ds')