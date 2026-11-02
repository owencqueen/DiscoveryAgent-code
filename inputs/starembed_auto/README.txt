# StarEmbed Classification Dataset - Train/Test Splits

This directory contains the training and test datasets for classification tasks using multi-band light curve data.

## Dataset Overview

```
DatasetDict({
    train: Dataset({
        features: ['sourceid', 'bands_data', 'item_id', 'start', 'freq', 'period', 'objid', 'csdr1_id', 'class_str'],
        num_rows: 29047
    })
    validation: Dataset({
        features: ['sourceid', 'bands_data', 'item_id', 'start', 'freq', 'period', 'objid', 'csdr1_id', 'class_str'],
        num_rows: 4150
    })
})
```

**Note:** There is also a validation split available in the `star_embed-validation.arrow` file.

---

## Dataset Structure

### Top-Level Features

| Feature | Type | Description |
|---------|------|-------------|
| `sourceid` | int64 | Unique source identifier |
| `bands_data` | dict | Multi-band photometric time series data (see below) |
| `item_id` | string | Item identifier |
| `start` | timestamp[us] | Start date/time of observations |
| `freq` | string | Sampling frequency (e.g., "1D" for daily) |
| `period` | float64 | Catalog period of the variable star |
| `objid` | string | Object identifier (cross-matched to Pan-STARRS1) |
| `csdr1_id` | string | Catalina Sky Survey Data Release 1 identifier |
| `class_str` | string | Classification label for the astronomical object |

### Multi-Band Data Structure (`bands_data`)

The `bands_data` field contains photometric data for three bands: **g**, **i**, and **r**. Each band has the following structure:

#### Per-Band Fields

| Field | Type | Description |
|-------|------|-------------|
| `target` | sequence of float64 | Magnitude measurements (brightness in AB system) |
| `mjd` | sequence of float64 | Modified Julian Date for each observation |
| `past_feat_dynamic_real` | sequence of float64 | Magnitude error (1-sigma uncertainty) |
| `feat_dynamic_real` | sequence of float64 | Time gap (delta_t) between consecutive observations |
| `length` | int64 | Number of observations in this band |

### Complete Feature Schema

```json
{
  "features": {
    "sourceid": {
      "dtype": "int64",
      "_type": "Value"
    },
    "bands_data": {
      "g": {
        "target": {
          "feature": {"dtype": "float64", "_type": "Value"},
          "_type": "Sequence"
        },
        "mjd": {
          "feature": {"dtype": "float64", "_type": "Value"},
          "_type": "Sequence"
        },
        "past_feat_dynamic_real": {
          "feature": {"dtype": "float64", "_type": "Value"},
          "_type": "Sequence"
        },
        "feat_dynamic_real": {
          "feature": {"dtype": "float64", "_type": "Value"},
          "_type": "Sequence"
        },
        "length": {
          "dtype": "int64",
          "_type": "Value"
        }
      },
      "i": { /* same structure as g */ },
      "r": { /* same structure as g */ }
    },
    "item_id": {"dtype": "string", "_type": "Value"},
    "start": {"dtype": "timestamp[us]", "_type": "Value"},
    "freq": {"dtype": "string", "_type": "Value"},
    "period": {"dtype": "float64", "_type": "Value"},
    "objid": {"dtype": "string", "_type": "Value"},
    "csdr1_id": {"dtype": "string", "_type": "Value"},
    "class_str": {"dtype": "string", "_type": "Value"}
  }
}
```

---

## Files in This Directory

- `star_embed-train-00000-of-00002.arrow` - Training data (part 1)
- `star_embed-train-00001-of-00002.arrow` - Training data (part 2)
- `star_embed-validation.arrow` - Validation data
- `README.txt` - This file

---

## Usage Notes

1. **Multi-band data**: Each object has observations in up to three photometric bands (g, i, r). Not all objects may have data in all bands.

2. **Time series structure**: The data is organized as irregular time series, with observations at different times captured in the `mjd` field.

3. **Classification labels**: The `class_str` field contains the astronomical classification for each source (e.g., variable star types).

4. **Data loading**: Use the Hugging Face `datasets` library to load this data:
   ```python
   from datasets import load_from_disk
   dataset = load_dataset("path/to/data")
   ```

---