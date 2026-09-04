class TagalogToBaybayinLocalTranslator {
  final Map<String, String> baseMap = {
    'a': 'ᜀ',
    'e': 'ᜁ',
    'i': '\u1717',
    'o': 'ᜂ',
    'u': '\u1718',
    'ba': 'ᜊ',
    'ka': 'ᜃ',
    'da': 'ᜇ',
    'ra': 'ᜍ',
    'ga': 'ᜄ',
    'ha': 'ᜑ',
    'la': 'ᜎ',
    'ma': 'ᜋ',
    'na': 'ᜈ',
    'nga': 'ᜅ',
    'pa': 'ᜉ',
    'sa': 'ᜐ',
    'ta': 'ᜆ',
    'wa': 'ᜏ',
    'ya': 'ᜌ',
  };

  final String kudlitE = '\u1715';
  final String kudlitI = '\u1712';
  final String kudlitU = '\u1716';
  final String kudlitO = '\u1713';
  final String virama = '\u1714';
  final String danda = '᜵';
  final String doubleDanda = '᜶';

  Map<String, dynamic> translate(String text) {
    if (text.trim().isEmpty) {
      return {'translated_text': '', 'confidence': 0.0};
    }

    final originalText = text.toLowerCase().trim();
    double confidence = 100.0;

    final nonNativeMatches = RegExp(r'[cfjqzvx]').allMatches(originalText);
    if (nonNativeMatches.isNotEmpty) {
      confidence -= nonNativeMatches.length * 15;
    }

    var workingText = originalText.replaceAll('ng', 'NG');

    if (workingText == 'mga') {
      return {'translated_text': 'ᜋᜄ', 'confidence': 100.0};
    }

    final pattern = RegExp(
      r'(NG[aeiou]|(?:[bkdrghlmnpstwry])?[aeiou])|(NG|[bkdrghlmnpstwry])|([aeiou])|(\s+)|(\.|\,)',
    );

    final buffer = StringBuffer();
    final matches = pattern.allMatches(workingText);

    for (final match in matches) {
      final cv = match.group(1);
      final c = match.group(2);
      final v = match.group(3);
      final space = match.group(4);
      final punct = match.group(5);

      if (space != null) {
        buffer.write(space);
        continue;
      }

      if (punct != null) {
        if (punct == '.') {
          buffer.write(doubleDanda);
        } else if (punct == ',') {
          buffer.write(danda);
        }
        continue;
      }

      final vowelToken = v ??
          (cv != null && cv.length == 1 && RegExp(r'^[aeiou]$').hasMatch(cv) ? cv : null);
      if (vowelToken != null) {
        buffer.write(baseMap[vowelToken] ?? '');
        continue;
      }

      if (cv != null) {
        final vowelPart = cv.substring(cv.length - 1);
        final consPart = cv.substring(0, cv.length - 1);

        String key;
        if (consPart == 'r') {
          key = 'ra';
        } else if (consPart == 'NG') {
          key = 'nga';
        } else {
          key = '${consPart}a';
        }

        final base = baseMap[key] ?? '';

        if (vowelPart == 'e') {
          buffer.write(base + kudlitE);
        } else if (vowelPart == 'i') {
          buffer.write(base + kudlitI);
        } else if (vowelPart == 'o') {
          buffer.write(base + kudlitO);
        } else if (vowelPart == 'u') {
          buffer.write(base + kudlitU);
        } else {
          buffer.write(base);
        }
      } else if (c != null) {
        String key;
        if (c == 'r') {
          key = 'ra';
        } else if (c == 'NG') {
          key = 'nga';
        } else {
          key = '${c}a';
        }

        final base = baseMap[key] ?? '';
        if (base.isNotEmpty) {
          buffer.write(base + virama);
        }
      }
    }

    return {
      'translated_text': buffer.toString(),
      'confidence': confidence < 0 ? 0.0 : confidence,
    };
  }
}
