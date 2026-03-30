# Performance Analysis

## Methodology

Profiling was performed using time measurements and cProfile.

## Metrics

* Average processing time: 0.5–0.8 seconds

## Hotspots

* predict_top3
* preprocess_to_mnist
* cv2.imdecode

## Conclusion

Model inference is the most time-consuming operation.
