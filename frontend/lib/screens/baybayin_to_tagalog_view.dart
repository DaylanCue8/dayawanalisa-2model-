import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';
import '../widgets/image_cropper_widget.dart';
import '../widgets/evaluation_modal.dart';

/// Handles the "Baybayin to Tagalog" mode: capture/upload a photo, crop it,
/// send it for translation, and show the result. Fully self-contained —
/// owns its own state, independent of the text-translation mode.
class BaybayinToTagalogView extends StatefulWidget {
  const BaybayinToTagalogView({super.key});

  @override
  State<BaybayinToTagalogView> createState() => _BaybayinToTagalogViewState();
}

class _BaybayinToTagalogViewState extends State<BaybayinToTagalogView> {
  final ApiService _apiService = ApiService();
  final ImagePicker _picker = ImagePicker();

  String _translatedResult = "Result will appear here";
  bool _isLoading = false;
  Uint8List? _webImage;

  Future<void> _processCroppedImage(Uint8List imageBytes) async {
    setState(() {
      _isLoading = true;
      _translatedResult = 'Processing Image...';
    });

    final response = await _apiService.uploadAndTranslateDetailed(
      null,
      'Baybayin to Tagalog',
      imageBytes: imageBytes,
    );

    if (!mounted) return;

    setState(() {
      _isLoading = false;
      if (response != null) {
        _translatedResult = response['translated_text'] ?? 'No result';

        String status = response['status']?.toString().toLowerCase() ?? '';
        if (status == 'success' || status == 'low_confidence') {
          Future.delayed(const Duration(milliseconds: 500), () {
            if (mounted) _showEvaluation(response);
          });
        } else if (status == 'no_characters' || _translatedResult.isEmpty) {
          _translatedResult = 'No Baybayin letters found. Try a clearer crop.';
        }
      } else {
        _translatedResult = 'Error: Connection Failed';
      }
    });
  }

  Future<void> _selectAndCropImage(ImageSource source) async {
    final XFile? photo = await _picker.pickImage(
      source: source,
      imageQuality: 80,
      maxWidth: 700,
      maxHeight: 700,
    );
    if (photo == null) return;

    final bytes = await photo.readAsBytes();
    if (!mounted) return;

    final Uint8List? croppedBytes = await Navigator.of(context).push<Uint8List>(
      MaterialPageRoute(
        builder: (_) => ImageCropperScreen(imageData: bytes),
      ),
    );

    if (croppedBytes == null) return;

    setState(() {
      _isLoading = true;
      _translatedResult = 'Processing Image...';
      // Send the raw cropped photo as-is — no client-side binarization.
      // app.py's Otsu-based pipeline is the single source of truth for
      // binarization and is tuned to handle lighting/shadows itself.
      _webImage = croppedBytes;
    });

    await _processCroppedImage(croppedBytes);
  }

  Future<void> _uploadFromGallery() async {
    await _selectAndCropImage(ImageSource.gallery);
  }

  Future<void> _captureFromCamera() async {
    await _selectAndCropImage(ImageSource.camera);
  }

  void _showEvaluation(Map<String, dynamic> data) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => EvaluationModal(
        detections: data['individual_detections'] ?? [],
        averageConfidence: (data['confidence'] as num).toDouble(),
        translatedText: data['translated_text'] ?? "",
        sessionId: data['session_id'] ?? 0,
      ),
    );
  }

  Widget _buildImageDisplay() {
    return Stack(
      children: [
        Center(
          child: _webImage != null
              ? Image.memory(_webImage!, fit: BoxFit.contain)
              : Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: const [
                      Icon(Icons.document_scanner, size: 64, color: Colors.brown),
                      SizedBox(height: 12),
                      Text(
                        "Upload or scan a document containing Baybayin scripts to transcribe",
                        textAlign: TextAlign.center,
                        style: TextStyle(color: Colors.grey, fontSize: 14),
                      ),
                    ],
                  ),
                ),
        ),
        if (_isLoading)
          Positioned.fill(
            child: ColoredBox(
              color: Colors.black.withOpacity(0.24),
              child: Center(
                child: Card(
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: const [
                        CircularProgressIndicator(color: Colors.brown),
                        SizedBox(height: 12),
                        Text(
                          "Processing text algorithm...",
                          style: TextStyle(fontWeight: FontWeight.w500),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          )
        else if (_webImage != null)
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: const EdgeInsets.all(12),
              color: Colors.black54,
              child: Text(
                _translatedResult,
                style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildUploadWidget() {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.brown.withOpacity(0.1),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.photo_library, size: 28, color: Colors.brown),
        ),
        const SizedBox(height: 8),
        const Text("Gallery", style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: Colors.black87)),
      ],
    );
  }

  Widget _buildCameraWidget() {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.brown.withOpacity(0.1),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.camera_alt, size: 28, color: Colors.brown),
        ),
        const SizedBox(height: 8),
        const Text("Camera", style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: Colors.black87)),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 20),
            decoration: BoxDecoration(
              color: const Color(0xFFF5F5F5),
              borderRadius: BorderRadius.circular(15),
            ),
            clipBehavior: Clip.antiAlias,
            child: _buildImageDisplay(),
          ),
        ),
        const SizedBox(height: 30),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 30),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              GestureDetector(onTap: _uploadFromGallery, child: _buildUploadWidget()),
              const SizedBox(width: 40),
              GestureDetector(onTap: _captureFromCamera, child: _buildCameraWidget()),
            ],
          ),
        ),
      ],
    );
  }
}