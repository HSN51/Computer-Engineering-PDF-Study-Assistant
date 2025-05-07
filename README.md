# Computer Engineering PDF Study Assistant

Bu proje, yüklediğiniz PDF ders notları üzerinden otomatik özet, flashcard, quiz üretimi, döküman içi arama ve sınav/pratik odaklı akıllı soru-cevap sunan modern bir Chatbot uygulamasıdır.

## Özellikler
- 📄 PDF yükleme ve otomatik metin çıkarma
- 🤖 Sınav/pratik odaklı, teknik ve kısa cevaplar
- 📝 Otomatik özet üretimi (kısa/uzun)
- 🃏 Flashcard (soru-cevap kartı) üretimi
- 📝 Quiz (çoktan seçmeli) soru üretimi
- 🔍 Döküman içi arama ve sayfa atlama
- 💡 Akıllı chunk seçimi (embedding tabanlı)
- 🎨 Modern ve responsive arayüz
- 💬 Sohbet geçmişi ve kaynak gösterimi

## Kurulum
1. **Depoyu klonlayın:**
   ```bash
   git clone <repo-url>
   cd <repo-klasörü>
   ```
2. **Ortamı oluşturun ve bağımlılıkları yükleyin:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **.env dosyasını oluşturun:**
   ```env
   OPENAI_API_KEY=sk-...
   ```
4. **Uygulamayı başlatın:**
   ```bash
   streamlit run app.py
   ```

## Kullanım
- Sol menüden PDF yükleyin.
- Otomatik özet, flashcard veya quiz üretmek için ilgili butonlara tıklayın.
- Döküman içi arama ile anahtar kelime arayın, ilgili sayfaya atlayın.
- Sohbet kutusundan teknik/sınav odaklı sorular sorun.

## Demo
> Demo videosu veya GIF ekleyin (ör: `demo.gif`)

## Ekran Görüntüsü
![Demo](demo.png)

## Katkı
Pull request ve issue açabilirsiniz.

## Lisans
MIT 