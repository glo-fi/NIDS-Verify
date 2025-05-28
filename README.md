# NIDS-Verify

This repo contains a simplified version of our work in exploring verifiable global constraints and adversarial training to improve NIDS model generalisation.

Due to time constraints, we don't recreate our full pipeline and results exactly. However, we provide a subset of our data alongside a simplified adversarial training process that targets only a single specification, our cross-dataset DoS Hulk specification. While this is simpler than those attempted in the paper, this exaggerates the impact of our adversarial training process and is likely a clearer demonstration.

## Installing Vehicle

Vehicle is easy to install via `pip`.

```pip install vehicle-lang```

The necessary commands will be added to your `$PATH`.

## Dataset

We provide a truncated version of our processed dataset, consisting of a subset of our benign data (from CIC IDS 2017) and our malicious data (from both CIC IDS 2017 (training) and CIC IDS 2018 (test)).

This consists of 42 features, 2 flow-level features and then 4 packet-level features for the first 10 packets: Protocol, TimeElapsed, Packet Directions (0 or 1), Packet Flags

Some code in the repo relates to our pipeline for feature extraction/calculation. This can be ignored.

## Adversarial training process

In this repo, we consider a simplified version of our training process, targeting only our . In the full paper, we adversarially train in order to satisfy to multiple specifications simultaneously.

The adversarial training process relies on a hyperparameter `alfa` to determine its strength. The correct value of `alfa` is difficult to know a priori, as it depends on the complexity of the specification, the adversarial training data and the shape of the model. We recommend running a broad hyperparameter search between, say, 0.5 and 0.000000005 and considering only models which score highly on the adversarial data. For this particular repo, we've set `alfa` to 5e-05.

The higher `alfa` is the *weaker* the adversarial training process is. It can be 'switched off' entirely by setting `alfa` to 1.0.

## Models

We provide two models: one which did not score highly on the adversarial data and one which did. The former of this does not satisfy our `propertyHulk` specification, while the latter does.

## Specification

Our specifications are provided in the `vehicle/global.vcl` file. These contain some additional conditions that we found to be necessary during our 'human-in-the-loop' specification development process.

The models first need to be converted into the ONNX format. We provide the code to do this for our two example models.