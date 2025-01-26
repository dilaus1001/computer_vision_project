import 'package:flutter/material.dart';
import '../models/detection_result.dart';

class DetectionProvider with ChangeNotifier {
  List<DetectionResult> _detections = [];
  bool _isProcessing = false;
  String? _errorMessage;
  
  // Getters to access our state
  List<DetectionResult> get detections => _detections;
  bool get isProcessing => _isProcessing;
  String? get errorMessage => _errorMessage;
  
  // Update detection results
  void updateDetections(List<DetectionResult> newDetections) {
    _detections = newDetections;
    _errorMessage = null;
    notifyListeners();  // This tells Flutter to rebuild widgets that depend on this data
  }
  
  // Set processing state
  void setProcessing(bool processing) {
    _isProcessing = processing;
    notifyListeners();
  }
  
  // Handle errors
  void setError(String message) {
    _errorMessage = message;
    _isProcessing = false;
    notifyListeners();
  }
  
  // Clear current state
  void clear() {
    _detections = [];
    _errorMessage = null;
    _isProcessing = false;
    notifyListeners();
  }
}