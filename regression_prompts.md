# Regression tests (manual on real Qwen/Whisper models)

## Text conversations
1. "how to be a sinior in aai و ممكن ياخد مني وقت قد اه لو فاقل وقت ممكن يعني"
2. "طب لو معايا مشاريع AI كبيرة بالفعل، هقلل المدة قد إيه عشان أبقى Senior؟"
3. "أنا مش بطلب مدة أعمل مشروع. أنا بسأل عن مسمى Senior وظيفيًا."
4. "مش عايز أشتغل على GPA، عايز أستخدم مشاريعي وخبرتي."
5. Change to a new unrelated topic, then return to AI engineer; observe whether wrong context leaks.

## Voice bilingual speech sample
Record at least 10 clips, one at a time, and write reference transcripts.
- "أنا يوسف and I want to be a Senior AI Engineer"
- "أنا شغال على Python و PyTorch وبعمل Deployment"
- "الموديل بيستخدم Machine Learning على Data من المشروع"
- A sentence spoken twice intentionally with silence in the middle.
- A 5-second silence after the sentence; expect zero added words.

Score separate metrics: Arabic words, English words preserved in Latin characters, duplications, latency to first provisional text, and total transcription time. Compare V9 and V10 on the same audio. No measured WER or improvement % is claimed here.
