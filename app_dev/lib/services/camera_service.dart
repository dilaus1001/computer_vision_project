import 'package:camera/camera.dart';

class CameraService {
  CameraController? controller;
  List<CameraDescription> cameras = [];

  Future<void> initialize() async {
    // Initialize the camera
    cameras = await availableCameras();
    if (cameras.isEmpty) return;

    controller = CameraController(
      cameras[0],  // Use the back camera
      ResolutionPreset.high,
      enableAudio: false,
    );

    await controller?.initialize();
  }

  Future<XFile?> takePicture() async {
    if (controller?.value.isInitialized ?? false) {
      return await controller?.takePicture();
    }
    return null;
  }

  void dispose() {
    controller?.dispose();
  }
}