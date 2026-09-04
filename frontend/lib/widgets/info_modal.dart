import 'package:flutter/material.dart';

class InfoModal extends StatelessWidget {
  const InfoModal({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: BoxConstraints(
        maxHeight: MediaQuery.of(context).size.height * 0.85,
      ),
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(25)),
      ),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Handle bar
            Center(
              child: Container(
                width: 50,
                height: 5,
                decoration: BoxDecoration(
                  color: Colors.grey[300],
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
            ),
            const SizedBox(height: 20),
            
            const Text(
              "About Dayaw",
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.brown),
            ),
            const SizedBox(height: 10),
            const Text(
              "Dayaw is a research project from Leyte Normal University (LNU) dedicated to preserving the ancient Filipino script through AI-assisted recognition.",
              style: TextStyle(fontSize: 15, color: Colors.black87),
            ),

            const Divider(height: 40),

            const Text(
              "How to Write for the AI",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.brown),
            ),
            const SizedBox(height: 10),
            const Text(
              "For the SVM + HOG model to recognize your handwriting accurately, please follow these visual guidelines:",
              style: TextStyle(fontSize: 14, color: Colors.black54),
            ),
            const SizedBox(height: 15),

            // Supported Characters Section
            Container(
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.brown[50],
                borderRadius: BorderRadius.circular(15),
              ),
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  const Text(
                    "Supported Characters",
                    style: TextStyle(fontWeight: FontWeight.bold, color: Colors.brown),
                  ),
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 10,
                    runSpacing: 10,
                    alignment: WrapAlignment.center,
                    children: _buildCharacterChips(),
                  ),
                  const Divider(height: 30, color: Colors.brown),
                  const Text(
                    "Vowel Markers (Kudlit)",
                    style: TextStyle(fontWeight: FontWeight.bold, color: Colors.brown),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    "Dot Above: E/I  •  Dot Below: O/U  •  Cross: No Vowel",
                    style: TextStyle(fontSize: 12, color: Colors.brown),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),

            const SizedBox(height: 20),
            const Text(
              "Best Practices:",
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const BulletPoint(text: "Use black ink on plain white paper."),
            const BulletPoint(text: "Keep characters separated (no touching)."),
            const BulletPoint(text: "Ensure dots/kudlits are clear and precise."),
            const BulletPoint(text: "Avoid shadows or glare in your photos."),
            
            const SizedBox(height: 30),
            
            // Final Action Button
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () => Navigator.pop(context),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.brown,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 15),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                child: const Text(
                  "Ipagpatuloy", 
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ),
            ),
            
            const SizedBox(height: 20),
            Center(
              child: Text(
                "DAYAW - Capstone Project © 2026",
                style: TextStyle(fontSize: 11, color: Colors.grey[400], letterSpacing: 1.2),
              ),
            ),
            const SizedBox(height: 10),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildCharacterChips() {
    final chars = ['ᜀ', 'ᜊ', 'ᜃ', 'ᜇ', 'ᜄ', 'ᜑ', 'ᜎ', 'ᜌ', 'ᜈ', 'ᜅ', 'ᜉ', 'ᜐ', 'ᜆ', 'ᜏ', 'ᜌ'];
    return chars.map((c) => Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 4,
            offset: const Offset(0, 2),
          )
        ],
      ),
      child: Text(c, style: const TextStyle(fontSize: 22, color: Colors.brown)),
    )).toList();
  }
}

class BulletPoint extends StatelessWidget {
  final String text;
  const BulletPoint({super.key, required this.text});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text("• ", style: TextStyle(fontWeight: FontWeight.bold, color: Colors.brown, fontSize: 18)),
          Expanded(child: Text(text, style: const TextStyle(fontSize: 14, height: 1.4))),
        ],
      ),
    );
  }
}