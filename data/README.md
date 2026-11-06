# Dataset files

The X-ray dataset is not stored in Git. Prepared splits remain in the KaggleHub cache or another user-configured location.

`dataset_manifest.csv` and `dataset_manifest_summary.json` describe the currently verified prepared split without containing image pixels. The CSV paths are relative to the prepared split root. Check the directory against these records before training:

```powershell
python src/data/validate_dataset.py --split-dir "C:\path\to\chest_xray_cnn_splits"
```

Do not overwrite the manifest when generating a different split; archive/version the new manifest and identify it with the corresponding experiment results.
