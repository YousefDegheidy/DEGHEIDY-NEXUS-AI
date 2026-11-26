# تقرير إصلاح QA74 Stage 3 — 24 نوفمبر 2026

تم استخراج الحزمة الأصلية QA74_SUPERVISED_T4_FULL_SOURCE، وتشغيل اختبارات الأصل: 194 passed, 13 subtests passed. بعد تعديل trainer وإضافة 10 اختبارات، شُغلت المجموعة: 204 passed, 13 subtests passed. بيئة CPU: Python 3.13.5، torch 2.10.0+cpu، CUDA=False. اختبار subprocess حقيقي فاشل عمدًا أثبت تسجيل stderr وكود الخروج. اختبارات القناع استعملت tokenizer محاكياً بإزاحات الأحرف، وشملت 140 مثالًا، ومحادثات متعددة الأدوار، ورفض وسوم الدور المحقونة، وإخفاء نص المستخدم والنظام، ورفض الطول الزائد. لا يوجد اختبار حقيقي للـHF tokenizer أو GPU: Hugging Face غير قابل للوصول من بيئة التنفيذ (DNS)، وUnsloth/Transformers غير مثبتين محليًا. لم يتم تدريب أي أوزان أو قياس دقة أخبار حية أو جودة 8/10. لا يوجد تأكيد أن عيب القناع هو السبب الفعلي للصورة، فالـtraceback الداخلي غير ظاهر. لا تغيير في التطبيق الأصلي أو بيانات التدريب.

مراجع خارجية: قالب Qwen3.5-4B في commit المذكور في الصورة، وتوثيق Unsloth لـQwen3.5 (Transformers v5، و16-bit LoRA):
- https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/chat_template.jinja
- https://unsloth.ai/docs/models/qwen3.5/fine-tune
