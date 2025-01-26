import 'dart:math' show Rectangle;

class DetectionResult {
  // Changed from List<double> to structured properties that clearly represent the box
  final double x;        // x coordinate of top-left corner
  final double y;        // y coordinate of top-left corner
  final double width;    // width of bounding box
  final double height;   // height of bounding box
  final double confidence;  // detection confidence score
  final String label;      // class label (e.g., "mole", "skin lesion")

  // Constructor that takes all required values
  DetectionResult({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    required this.confidence,
    required this.label,
  });

  // Helper method to get the bounding box as a Rectangle
  Rectangle<double> get boundingBox => Rectangle<double>(x, y, width, height);

  // Helper method to get the bounding box as a List (if needed for compatibility)
  List<double> get boundingBoxAsList => [x, y, width, height];

  // Helper method to get the center point of the box
  Map<String, double> get center => {
    'x': x + width / 2,
    'y': y + height / 2,
  };

  // Helper method to check if a point is inside the bounding box
  bool containsPoint(double px, double py) {
    return px >= x && px <= x + width && py >= y && py <= y + height;
  }

  // Create a detection from YOLO output format
  static DetectionResult fromYOLOOutput({
    required double centerX,
    required double centerY,
    required double width,
    required double height,
    required double confidence,
    String label = 'mole',  // Default label if not specified
  }) {
    // Convert from center coordinates to top-left coordinates
    final x = centerX - width / 2;
    final y = centerY - height / 2;

    return DetectionResult(
      x: x,
      y: y,
      width: width,
      height: height,
      confidence: confidence,
      label: label,
    );
  }

  // String representation for debugging
  @override
  String toString() {
    return 'DetectionResult(x: $x, y: $y, width: $width, height: $height, '
           'confidence: ${confidence.toStringAsFixed(3)}, label: $label)';
  }
}