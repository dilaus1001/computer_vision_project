import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:path/path.dart' as path;
import 'package:skc_app/models/detection_result.dart';

class ModelService {
  late final Interpreter _skinCancerInterpreter;
  late final Interpreter _yoloInterpreter;
  static const int TARGET_SIZE = 608;
  static const int SKC_TARGET_SIZE = 224;
  static const double CONFIDENCE_THRESHOLD = 0.5;
  static const double IOU_THRESHOLD = 0.5; // Matching Python's IOU threshold

  Future<void> initialize() async {
    try {
      final options = InterpreterOptions()..threads = 4;
      _yoloInterpreter = await Interpreter.fromAsset(
        'assets/yolo11n_float32.tflite',
        options: options,
      );
      _skinCancerInterpreter = await Interpreter.fromAsset(
        'assets/skc_model.tflite',
        options: options,
      );
      _skinCancerInterpreter.allocateTensors();
      debugPrint('Models initialized successfully');
    } catch (e) {
      debugPrint('Error initializing models: $e');
      rethrow;
    }
  }

  Future<Map<String, dynamic>> _preprocessImage(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    final image = img.decodeImage(bytes);
    if (image == null) throw Exception('Failed to decode image');

    final originalWidth = image.width;
    final originalHeight = image.height;
    
    // Calculate scaling factor as in Python
    final double ratio = TARGET_SIZE / max(originalWidth, originalHeight);
    final int newWidth = (originalWidth * ratio).round();
    final int newHeight = (originalHeight * ratio).round();

    // Resize using Lanczos interpolation like PIL
    final resized = img.copyResize(
      image,
      width: newWidth,
      height: newHeight,
      interpolation: img.Interpolation.cubic, // Best approximation of LANCZOS
    );

    // Create black canvas and calculate padding like Python
    final paddedImage = img.Image(width: TARGET_SIZE, height: TARGET_SIZE);
    final int dx = (TARGET_SIZE - newWidth) ~/ 2;
    final int dy = (TARGET_SIZE - newHeight) ~/ 2;

    // Paste resized image onto canvas pixel by pixel
    for (var y = 0; y < newHeight; y++) {
      for (var x = 0; x < newWidth; x++) {
        final pixel = resized.getPixel(x, y);
        paddedImage.setPixel(x + dx, y + dy, pixel);
      }
    }

    // Convert to normalized array like Python
    final inputArray = Float32List(TARGET_SIZE * TARGET_SIZE * 3);
    var pixelIndex = 0;
    for (var y = 0; y < TARGET_SIZE; y++) {
      for (var x = 0; x < TARGET_SIZE; x++) {
        final pixel = paddedImage.getPixel(x, y);
        inputArray[pixelIndex++] = pixel.r / 255.0;
        inputArray[pixelIndex++] = pixel.g / 255.0;
        inputArray[pixelIndex++] = pixel.b / 255.0;
      }
    }

    return {
      'input_array': inputArray,
      'scale_factors': {
        'original_size': {'width': originalWidth, 'height': originalHeight},
        'new_size': {'width': newWidth, 'height': newHeight},
        'padding': {'dx': dx, 'dy': dy},
        'ratio': ratio,
      }
    };
  }

  Future<List<DetectionResult>> detectMoles(File imageFile) async {
    try {
      final preprocessResult = await _preprocessImage(imageFile);
      final inputArray = preprocessResult['input_array'] as Float32List;
      final scaleFactors = preprocessResult['scale_factors'] as Map<String, dynamic>;
      
      final originalWidth = scaleFactors['original_size']['width'] as int;
      final originalHeight = scaleFactors['original_size']['height'] as int;
      final dx = scaleFactors['padding']['dx'] as int;
      final dy = scaleFactors['padding']['dy'] as int;
      final ratio = scaleFactors['ratio'] as double;

      // Reshape input array to match model's expected shape [1, 608, 608, 3]
      final inputShape = [1, TARGET_SIZE, TARGET_SIZE, 3];
      final inputTensor = List.generate(1, (_) => 
        List.generate(TARGET_SIZE, (_) => 
          List.generate(TARGET_SIZE, (_) => 
            List.filled(3, 0.0))));

      // Copy data into the properly shaped tensor
      var index = 0;
      for (var y = 0; y < TARGET_SIZE; y++) {
        for (var x = 0; x < TARGET_SIZE; x++) {
          for (var c = 0; c < 3; c++) {
            inputTensor[0][y][x][c] = inputArray[index++];
          }
        }
      }

      // Run model
      final outputBuffer = List.generate(
        1,
        (_) => List.generate(5, (_) => List.filled(7581, 0.0)),
      );
      _yoloInterpreter.run(inputTensor, outputBuffer);

      final results = <DetectionResult>[];

      // Process predictions like Python decode_predictions()
      for (var i = 0; i < 7581; i++) {
        final confidence = outputBuffer[0][4][i];
        if (confidence > CONFIDENCE_THRESHOLD) {
          // Get normalized coordinates (relative to 608x608)
          final centerX = outputBuffer[0][0][i];
          final centerY = outputBuffer[0][1][i];
          final width = outputBuffer[0][2][i];
          final height = outputBuffer[0][3][i];

          // Remove padding and scale back to original image space
          final x = (centerX * TARGET_SIZE - dx) / ratio;
          final y = (centerY * TARGET_SIZE - dy) / ratio;
          final w = (width * TARGET_SIZE) / ratio;
          final h = (height * TARGET_SIZE) / ratio;

          // Convert to corner coordinates
          final xmin = max(0, (x - w/2)).floor().toDouble();
          final ymin = max(0, (y - h/2)).floor().toDouble();
          final xmax = min(originalWidth, (x + w/2)).ceil().toDouble();
          final ymax = min(originalHeight, (y + h/2)).ceil().toDouble();

          results.add(DetectionResult(
            x: xmin,
            y: ymin,
            width: xmax - xmin,
            height: ymax - ymin,
            confidence: confidence,
            label: 'mole',
          ));
        }
      }

      // Apply NMS like Python
      if (results.isNotEmpty) {
        return _nonMaxSuppression(results, IOU_THRESHOLD);
      }

      return [];
    } catch (e) {
      debugPrint('Error in detectMoles: $e');
      rethrow;
    }
  }

  List<DetectionResult> _nonMaxSuppression(List<DetectionResult> boxes, double iouThreshold) {
    boxes.sort((a, b) => b.confidence.compareTo(a.confidence));
    final List<DetectionResult> selected = [];
    final Set<int> suppressed = {};

    for (var i = 0; i < boxes.length; i++) {
      if (suppressed.contains(i)) continue;
      
      selected.add(boxes[i]);
      
      for (var j = i + 1; j < boxes.length; j++) {
        if (suppressed.contains(j)) continue;
        
        if (_calculateIoU(boxes[i], boxes[j]) >= iouThreshold) {
          suppressed.add(j);
        }
      }
    }
    
    return selected;
  }

  double _calculateIoU(DetectionResult box1, DetectionResult box2) {
    final x1 = max(box1.x, box2.x);
    final y1 = max(box1.y, box2.y);
    final x2 = min(box1.x + box1.width, box2.x + box2.width);
    final y2 = min(box1.y + box1.height, box2.y + box2.height);

    if (x2 <= x1 || y2 <= y1) return 0.0;

    final intersection = (x2 - x1) * (y2 - y1);
    final box1Area = box1.width * box1.height;
    final box2Area = box2.width * box2.height;
    final union = box1Area + box2Area - intersection;

    return intersection / union;
  }

  Future<double> analyzeMole(File imageFile, DetectionResult detection) async {
    try {
      final bytes = await imageFile.readAsBytes();
      final image = img.decodeImage(bytes);
      if (image == null) throw Exception('Failed to decode image');

      // Crop the detected region
      final cropped = img.copyCrop(
        image,
        x: detection.x.round(),
        y: detection.y.round(),
        width: detection.width.round(),
        height: detection.height.round(),
      );

      // Resize to 224x224 using Lanczos interpolation
      final resized = img.copyResize(
        cropped,
        width: SKC_TARGET_SIZE,
        height: SKC_TARGET_SIZE,
        interpolation: img.Interpolation.cubic,
      );

      // Prepare input tensor like Python
      var input = List.generate(
        1,
        (_) => List.generate(
          SKC_TARGET_SIZE,
          (_) => List.generate(
            SKC_TARGET_SIZE,
            (_) => List.filled(3, 0.0),
          ),
        ),
      );

      // Normalize pixels
      for (var y = 0; y < SKC_TARGET_SIZE; y++) {
        for (var x = 0; x < SKC_TARGET_SIZE; x++) {
          final pixel = resized.getPixel(x, y);
          input[0][y][x][0] = pixel.r / 255.0;
          input[0][y][x][1] = pixel.g / 255.0;
          input[0][y][x][2] = pixel.b / 255.0;
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