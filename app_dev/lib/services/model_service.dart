import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart' as path;
import 'package:skc_app/models/detection_result.dart';

class ModelService {
  late final Interpreter _skinCancerInterpreter;
  late final Interpreter _yoloInterpreter;
  static const int TARGET_SIZE = 608;
  static const int SKC_TARGET_SIZE = 224;
  static const double CONFIDENCE_THRESHOLD = 0.5;

  Future<void> initialize() async {
    try {
      final options = InterpreterOptions()..threads = 4;
      _yoloInterpreter = await Interpreter.fromAsset('assets/yolo11n_float32.tflite', options: options);
      _skinCancerInterpreter = await Interpreter.fromAsset('assets/skc_model.tflite', options: options);
      _skinCancerInterpreter.allocateTensors();
      debugPrint('Models initialized successfully');
    } catch (e) {
      debugPrint('Error initializing models: $e');
      rethrow;
    }
  }

  Future<Float32List> _preprocessImage(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    final image = img.decodeImage(bytes);
    if (image == null) throw Exception('Failed to decode image');

    final resized = img.copyResize(image, width: TARGET_SIZE, height: TARGET_SIZE);
    final inputArray = Float32List(TARGET_SIZE * TARGET_SIZE * 3);
    var pixelIndex = 0;

    for (var y = 0; y < TARGET_SIZE; y++) {
      for (var x = 0; x < TARGET_SIZE; x++) {
        final pixel = resized.getPixel(x, y);
        inputArray[pixelIndex++] = pixel.r / 255.0;
        inputArray[pixelIndex++] = pixel.g / 255.0;
        inputArray[pixelIndex++] = pixel.b / 255.0;
      }
    }

    return inputArray;
  }

  double calculateIoU(DetectionResult box1, DetectionResult box2) {
    double x1 = max(box1.x, box2.x);
    double y1 = max(box1.y, box2.y);
    double x2 = min(box1.x + box1.width, box2.x + box2.width);
    double y2 = min(box1.y + box1.height, box2.y + box2.height);

    double intersectionArea = max(0, x2 - x1) * max(0, y2 - y1);
    double box1Area = box1.width * box1.height;
    double box2Area = box2.width * box2.height;
    double unionArea = box1Area + box2Area - intersectionArea;

    return intersectionArea / unionArea;
  }

  List<DetectionResult> mergeDetections(List<DetectionResult> detections, double iouThreshold) {
    List<DetectionResult> filteredResults = [];
    detections.sort((a, b) => b.confidence.compareTo(a.confidence)); // Sort by confidence

    while (detections.isNotEmpty) {
      final bestDetection = detections.removeAt(0);
      filteredResults.add(bestDetection);

      detections = detections.where((test) {
        double iou = calculateIoU(bestDetection, test);
        return iou < iouThreshold;
      }).toList();
    }

    return filteredResults;
  }

  Future<List<DetectionResult>> detectMoles(File imageFile) async {
      try {
        final inputImage = await _preprocessImage(imageFile);
        final inputTensor = inputImage.reshape([1, TARGET_SIZE, TARGET_SIZE, 3]);
        
        final outputBuffer = List.generate(1, (_) => List.generate(5, (_) => List.filled(7581, 0.0)));
        _yoloInterpreter.run(inputTensor, outputBuffer);

        final results = <DetectionResult>[];
        final boxes = outputBuffer[0][0];
        final scores = outputBuffer[0][1];
        
        debugPrint('Processing ${boxes.length} detections');
        
        // Process each prediction
        for (var i = 0; i < 7581; i++) {
          final confidence = outputBuffer[0][4][i]; // Confidence score
          if (confidence > CONFIDENCE_THRESHOLD) {
            final centerX = outputBuffer[0][0][i] * TARGET_SIZE;
            final centerY = outputBuffer[0][1][i] * TARGET_SIZE;
            final width = outputBuffer[0][2][i] * TARGET_SIZE;
            final height = outputBuffer[0][3][i] * TARGET_SIZE;

            final x1 = centerX - width / 2;
            final y1 = centerY - height / 2;
            final x2 = centerX + width / 2;
            final y2 = centerY + height / 2;

            debugPrint('Detection: ($x1, $y1, $x2, $y2) Confidence: $confidence');

            results.add(DetectionResult(
              x: x1,
              y: y1,
              width: x2 - x1,
              height: y2 - y1,
              confidence: confidence,
              label: 'mole',
            ));
          }
        }


        debugPrint('Found ${results.length} detections');
        return results;
      } catch (e) {
        debugPrint('Error: $e');
        rethrow;
      }
    }

  Future<double> analyzeMole(File imageFile, DetectionResult detection) async {
    try {
      debugPrint('Analyzing mole: $detection');
      final bytes = await imageFile.readAsBytes();
      final image = img.decodeImage(bytes);
      if (image == null) throw Exception('Failed to decode image');

      final bbox = detection.boundingBox;
      if (bbox.width <= 0 || bbox.height <= 0) {
        throw Exception('Invalid bounding box dimensions: $bbox');
      }

      final cropped = img.copyCrop(
        image,
        x: bbox.left.round().clamp(0, image.width - 1),
        y: bbox.top.round().clamp(0, image.height - 1),
        width: bbox.width.round().clamp(1, image.width - bbox.left.round()),
        height: bbox.height.round().clamp(1, image.height - bbox.top.round()),
      );

      final resized = img.copyResize(cropped, width: SKC_TARGET_SIZE, height: SKC_TARGET_SIZE);
      debugPrint('Resized image dimensions: ${resized.width}x${resized.height}');

      // Prepare input data as per TFLite format
      var input = List.generate(1, (_) => 
        List.generate(SKC_TARGET_SIZE, (_) => 
          List.generate(SKC_TARGET_SIZE, (_) => 
            List.filled(3, 0.0))));

      for (var y = 0; y < SKC_TARGET_SIZE; y++) {
        for (var x = 0; x < SKC_TARGET_SIZE; x++) {
          final pixel = resized.getPixel(x, y);
          input[0][y][x][0] = pixel.r.toDouble() / 255.0;
          input[0][y][x][1] = pixel.g.toDouble() / 255.0;
          input[0][y][x][2] = pixel.b.toDouble() / 255.0;
        }
      }

      var output = List.generate(1, (_) => List.filled(1, 0.0));
      _skinCancerInterpreter.run(input, output);
      
      return output[0][0];
    } catch (e) {
      debugPrint('Error analyzing mole: $e');
      rethrow;
    }
  }


    void dispose() {
      _yoloInterpreter.close();
      _skinCancerInterpreter.close();
    }
}