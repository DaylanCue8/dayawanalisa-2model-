import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

/// Custom camera capture screen (replaces the OS camera app for this
/// flow) so we can force a high resolution preset and lock focus/
/// exposure before capture - neither of which is possible when going
/// through image_picker's ImageSource.camera, since that hands control
/// entirely to the OS camera app.
///
/// The white guide box overlay is NOT just decorative - after capture,
/// the photo is orientation-normalized and auto-cropped to exactly that
/// box's area before being returned, so what the user framed is what
/// they get.
///
/// Returns the captured, cropped JPEG bytes via Navigator.pop, or null
/// if the user backs out without capturing.
class CameraCaptureScreen extends StatefulWidget {
  const CameraCaptureScreen({super.key});

  @override
  State<CameraCaptureScreen> createState() => _CameraCaptureScreenState();
}

class _CameraCaptureScreenState extends State<CameraCaptureScreen> {
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;
  bool _isCapturing = false;
  Offset? _focusPoint;
  String? _error;

  // Guide box size as fractions of the screen - MUST match the
  // FractionallySizedBox values in the overlay below, since these
  // same fractions are applied directly to the captured image to crop
  // it to what the box outlined.
  static const double _guideWidthFactor = 0.85;
  static const double _guideHeightFactor = 0.5;

  @override
  void initState() {
    super.initState();
    _setupCamera();
  }

  Future<void> _setupCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _error = 'No camera found on this device.');
        return;
      }
      final backCamera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      // veryHigh gives strong detail for thin strokes without the
      // file-size/processing cost of forcing the absolute sensor max.
      // Bump to ResolutionPreset.max if strokes are still breaking up
      // in the backend pipeline after testing this.
      final controller = CameraController(
        backCamera,
        ResolutionPreset.veryHigh,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      _controller = controller;
      _initializeControllerFuture = controller.initialize().then((_) async {
        try {
          await controller.setFocusMode(FocusMode.auto);
          await controller.setExposureMode(ExposureMode.auto);
        } catch (_) {
          // Some devices/plugin versions don't support manual focus
          // mode changes - safe to ignore, autofocus still runs.
        }
        if (mounted) setState(() {});
      });
      setState(() {});
    } catch (e) {
      setState(() => _error = 'Failed to start camera: $e');
    }
  }

  /// Tap-to-focus: locks focus and exposure at the tapped point so the
  /// shot is sharp before capture, instead of relying on whatever the
  /// continuous autofocus happened to settle on.
  Future<void> _onTapToFocus(TapDownDetails details, BoxConstraints constraints) async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;

    final normalized = Offset(
      details.localPosition.dx / constraints.maxWidth,
      details.localPosition.dy / constraints.maxHeight,
    );
    setState(() => _focusPoint = details.localPosition);

    try {
      await controller.setFocusPoint(normalized);
      await controller.setExposurePoint(normalized);
      await controller.setFocusMode(FocusMode.locked);
    } catch (_) {
      // Not all devices support point focus - ignore and fall back to
      // whatever focus the camera already has.
    }
  }

  /// Bakes EXIF orientation into the pixels (so the image is upright,
  /// matching what the user saw in the preview), then crops to the
  /// guide box's fractional area. Falls back to the raw bytes if
  /// decoding fails for any reason, rather than losing the photo.
  Uint8List _cropToGuideBox(Uint8List rawBytes) {
    final decoded = img.decodeImage(rawBytes);
    if (decoded == null) return rawBytes;

    final oriented = img.bakeOrientation(decoded);

    final cropWidth = (oriented.width * _guideWidthFactor).round();
    final cropHeight = (oriented.height * _guideHeightFactor).round();
    final x = ((oriented.width - cropWidth) / 2).round();
    final y = ((oriented.height - cropHeight) / 2).round();

    final cropped = img.copyCrop(
      oriented,
      x: x,
      y: y,
      width: cropWidth,
      height: cropHeight,
    );

    return Uint8List.fromList(img.encodeJpg(cropped, quality: 95));
  }

  Future<void> _capture() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized || _isCapturing) return;

    setState(() => _isCapturing = true);
    try {
      final XFile file = await controller.takePicture();
      final rawBytes = await file.readAsBytes();
      final croppedBytes = _cropToGuideBox(rawBytes);
      if (!mounted) return;
      Navigator.of(context).pop<Uint8List>(croppedBytes);
    } catch (e) {
      if (mounted) {
        setState(() => _isCapturing = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Capture failed: $e')),
        );
      }
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
        backgroundColor: Colors.black,
        appBar: AppBar(title: const Text('Camera')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Text(
              _error!,
              style: const TextStyle(color: Colors.white),
              textAlign: TextAlign.center,
            ),
          ),
        ),
      );
    }

    final controller = _controller;
    if (controller == null || _initializeControllerFuture == null) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator(color: Colors.white)),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: FutureBuilder<void>(
          future: _initializeControllerFuture,
          builder: (context, snapshot) {
            if (snapshot.connectionState != ConnectionState.done) {
              return const Center(child: CircularProgressIndicator(color: Colors.white));
            }
            return LayoutBuilder(
              builder: (context, constraints) {
                return Stack(
                  fit: StackFit.expand,
                  children: [
                    GestureDetector(
                      onTapDown: (details) => _onTapToFocus(details, constraints),
                      child: CameraPreview(controller),
                    ),
                    // Framing guide - this is now the ACTUAL crop
                    // boundary, applied to the captured photo right
                    // after takePicture(). Keep _guideWidthFactor /
                    // _guideHeightFactor above in sync with these
                    // FractionallySizedBox values if you change either.
                    IgnorePointer(
                      child: Center(
                        child: FractionallySizedBox(
                          widthFactor: _guideWidthFactor,
                          heightFactor: _guideHeightFactor,
                          child: Container(
                            decoration: BoxDecoration(
                              border: Border.all(color: Colors.white70, width: 2),
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                        ),
                      ),
                    ),
                    if (_focusPoint != null)
                      Positioned(
                        left: _focusPoint!.dx - 20,
                        top: _focusPoint!.dy - 20,
                        child: IgnorePointer(
                          child: Container(
                            width: 40,
                            height: 40,
                            decoration: BoxDecoration(
                              border: Border.all(color: Colors.yellow, width: 2),
                              shape: BoxShape.circle,
                            ),
                          ),
                        ),
                      ),
                    Positioned(
                      top: 8,
                      left: 8,
                      child: IconButton(
                        icon: const Icon(Icons.close, color: Colors.white, size: 28),
                        onPressed: () => Navigator.of(context).pop(),
                      ),
                    ),
                    Positioned(
                      bottom: 24,
                      left: 0,
                      right: 0,
                      child: Center(
                        child: GestureDetector(
                          onTap: _isCapturing ? null : _capture,
                          child: Container(
                            width: 72,
                            height: 72,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              border: Border.all(color: Colors.white, width: 4),
                              color: _isCapturing ? Colors.grey : Colors.white24,
                            ),
                            child: _isCapturing
                                ? const Padding(
                                    padding: EdgeInsets.all(20),
                                    child: CircularProgressIndicator(color: Colors.white),
                                  )
                                : null,
                          ),
                        ),
                      ),
                    ),
                  ],
                );
              },
            );
          },
        ),
      ),
    );
  }
}