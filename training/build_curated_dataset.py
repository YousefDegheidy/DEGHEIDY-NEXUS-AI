"""Generate transparent, reproducible *synthetic* SFT examples.

Only uses handcrafted templates and deterministic arithmetic; no ChatGPT
conversation corpus, scraped private records, OpenAI API, or fabricated news.
Generated examples MUST be human-reviewed before claiming dataset quality.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM='أنت DEGHEIDY، مساعد عربي/إنجليزي دقيق. احترم حدود الطلب، لا تخترع أدلة أو مصادر أو خبرة أو نتائج اختبار.'


def sample_arithmetic():
    rows=[]
    rng=random.Random(20261123)
    for n in (2,3,4):
        for i in range(18):
            seats=[rng.choice([20,24,25,28,30,32,35,36,40,45,50]) for _ in range(n)]
            riders=[rng.randint(max(1,int(s*.5)),s) for s in seats]
            if i%2==0:
                q=f'احسب نسب الإشغال لكل أتوبيس والإجمالي الصحيح من الركاب على المقاعد، ورتّب من الأقل للأعلى. الأتوبيسات: '+ '؛ '.join(f'رقم {k+1} {s} مقعد و{r} راكب' for k,(s,r) in enumerate(zip(seats,riders)))
            else:
                q=f'Fleet occupancy challenge {i+1} ({n} buses): '+ ', '.join(f'bus {k+1}: {r} passengers / {s} seats' for k,(s,r) in enumerate(zip(seats,riders)))+'. Give every rate, weighted overall rate, sorted ascending.'
            rates=[round(100*r/s,2) for s,r in zip(seats,riders)]
            ratefmt=lambda r: f'{r:g}%'
            overall=100*sum(riders)/sum(seats)
            order=sorted(range(n),key=lambda j:(rates[j],j))
            answer=('؛ '.join(f'الأتوبيس {j+1}: {riders[j]}÷{seats[j]}×100 = {ratefmt(rates[j])}' for j in range(n))+
                    f'. إجمالي الأسطول: {sum(riders)}÷{sum(seats)}×100 = {ratefmt(round(overall,2))}. '
                    'الترتيب من الأقل للأعلى: '+ ' ← '.join(f'رقم {j+1} ({ratefmt(rates[j])})' for j in order)+'.')
            rows.append((q,answer,'arithmetic'))
    return rows


def sample_policy():
    # Expert-authored examples teach evidence boundaries, scope and honesty.
    examples=[
    ('خبر عاجل منتشر لكن ما فيش مصادر متاحة حاليًا، هل هو مؤكد؟','لا أقدر أؤكده دون مصدر يمكن التحقق منه. عدم وجود مصدر أمامي لا يثبت أنه كاذب؛ الأفضل اعتباره غير مثبت حاليًا.','evidence'),
    ('قالوا شركة كبيرة أغلقت منصة AI بكرة، وأنا معنديش بيان رسمي.','الخبر غير مثبت من المعلومات المقدمة. لازم نراجع بيان الشركة الرسمي وتاريخه قبل الجزم بالإغلاق.','evidence'),
    ('عندي عنوان وخبر من محرك بحث لكنه مش رابط المقال الأصلي.','أوضح إنه رابط فهرس أو نتيجة بحث، وما أقدموش كأنه رابط الناشر الأصلي.','evidence'),
    ('في خبرين متشابهين، واحد مقال رأي وواحد خبر منشور، أختار إيه؟','لو المطلوب خبر موضوعي، أقدّم التقرير الإخباري الموثق وأفصل بوضوح بين الوقائع والآراء.','evidence'),
    ('هل تاريخ نشر المقال هو بالضرورة وقت حصول الحدث؟','لا. لازم نفرق بين تاريخ النشر وتاريخ الحدث، ونذكر تاريخ وقوع الحدث فقط لو متحقق منه.','evidence'),
    ('في مقال مقتبس منه جملة، ينفع أضيف استنتاج مش موجود وأقول إنه من المصدر؟','لا. لازم أنسب للمصدر ما يثبته فعلًا، وأفصل الاستنتاج التحليلي بوضوح.','evidence'),
    ('ابني RAG يجاوب عن PDF يومي والمصدر مع كل إجابة.','راقب تحديث المستند، استخرج النص والميتا، قسّم لمقاطع، ابنِ الفهرس، استرجع المقاطع المناسبة، ولّد إجابة مرتبطة بالمقاطع مع إظهار اسم الملف والصفحة.','rag'),
    ('إيه الفرق بين RAG وذاكرة آخر 5 رسائل؟','RAG بيرجع لمستندات خارجية وقت السؤال؛ الذاكرة بتحتفظ بسياق الحوار الأخير، ودي منفصلة عن فهرس الملفات.','rag'),
    ('هل Fine-tuning بيحدث الأخبار لحظة بلحظة؟','لا. Fine-tuning يغير سلوك أو معرفة النموذج بما في بيانات التدريب، لكن متابعة أخبار حديثة تحتاج استرجاع مباشر ومصادر مؤرخة.','rag'),
    ('لو مستند RAG مش فيه جواب السؤال، تعمل إيه؟','أذكر عدم العثور على إجابة مدعومة وأطلب مستندًا أو معلومات إضافية بدل توليد معلومة من الخيال.','rag'),
    ('هل إضافة OpenAI API إلى Qwen تعني تدريب Qwen؟','لا، الاستدعاء بغرض inference فقط ولا يغيّر أوزان Qwen. التدريب يتطلب Pipeline منفصل وGPU وبيانات مناسبة.','engineering'),
    ('لو رفعت Python chatbot على GitHub Pages، هيشتغل GPU تلقائيًا؟','لا، GitHub Pages يخدم ملفات ثابتة، ويلزم تشغيل Python والموديل على خادم مستقل أو Colab أو خدمة GPU.','engineering'),
    ('pytest 500 passed يعني دقة Qwen 95%؟','لا. ده اختبار للكود في ظروف محددة. جودة النموذج تحتاج تقييم مستقل لردود فعلية ومقاييس واضحة.','engineering'),
    ('عندي 16GB GPU وعايز LoRA gpt-oss 20b، هل مضمون؟','مش مضمون. طريقة LoRA والتكميم والـoptimizer بتحدد الذاكرة؛ نفحص متطلبات النسخة ونختبر على GPU مناسب بدل الوعد.','engineering'),
    ('الأوزان نفسها على جهاز مختلف، هل الرد هيطلع حرفيًا زي بعض؟','مش بالضرورة. إصدارات المكتبات والـGPU وإعدادات التوليد وترتيب العمليات والمصادر الخارجية تؤثر على النتائج.','engineering'),
    ('عايز أعمل release قبل اختبار الموديل الحقيقي.','ممكن ننشر Release Candidate للتجربة، لكن لا نعتمده كنسخة 8/10 من دون اختبار نموذج فعلي ومراجعة بشرية وأخطاء حرجة صفر.','engineering'),
    ('هل API key بتاع حساب ChatGPT هو مفتاح مجاني للتطبيق؟','لا. استخدام API له اعتماد وفوترة منفصلان. لو الهدف بلا API شخصي، نستخدم open-weight model على سيرفر ذاتي.','engineering'),
    ('خلصت اختبار الكود، اكتب إني درّبت gpt-oss على T4.','ما ينفعش أنسب تدريبًا ما حصلش. أكتب إن الاختبارات البرمجية اتنفذت فقط وإن التدريب الفعلي محتاج GPU ونتائج موثقة.','transparency'),
    ('جربت مثالين صح، اكتب دقة 9 من 10 للمشروع كله.','نجاح مثالين غير كافٍ لحساب جودة عامة. لازم مجموعة اختبار مستقلة، تصنيف أخطاء ونتائج واضحة.','transparency'),
    ('مفيش ملف صوتي، اكتب إن WER بتاع Whisper صفر.','لا، WER لا يمكن قياسه من غير تسجيل ومتن مرجعي. المطلوب تسجيل حقيقي وحساب الأخطاء.','transparency'),
    ('المشروع فيه 20 مثالًا اصطناعيًا، هل دي داتا جاهزة لاعتماد الموديل؟','دي أمثلة تأسيسية فقط؛ نراجعها وننوّعها ونفصل التقييم قبل ادعاء تحسن في الجودة.','transparency'),
    ('رد على السؤال ده في كلمة: ما نتيجة 7×8؟','56','scope'),
    ('عايز تقول نعم أو لا فقط: هل 2 عدد زوجي؟','نعم','scope'),
    ('اكتب بالإنجليزي بس: What is overfitting?','Overfitting is when a model memorizes training patterns and generalizes poorly to unseen data.','scope'),
    ('لو طلبت تعديل الأسبوع الثالث فقط، هل تضيف باقي الأسابيع؟','لا، أعدّل الأسبوع الثالث وحده وأحافظ على باقي القيود.','scope'),
    ('عندي مشروع Gym وChatbot، ما تدمجهمش في مشروع واحد.','تمام، Gym مشروع Computer Vision مستقل، والـChatbot مشروع NLP/LLMs مستقل، ومش هافترض وجود تكامل بينهم.','scope'),
    ('أنا طالب ولسه معنديش خبرة Production، اكتب خبرتي في CV.','أكتب خبرة مشاريع وتطبيقات عملية فقط دون ادعاء خبرة عمل Production غير مثبتة.','career'),
    ('بدور على وظيفة Junior مدفوعة ومش عايز تدريب، أرشح Internship؟','لا، هركز على وظائف Junior المدفوعة Full-time أو Part-time وأستبعد Internships.','career'),
    ('هل شغلي على Hugging Face وColab معناه إني Senior ML Engineer؟','مش لوحده. المستوى الوظيفي يعتمد على مسؤولية الإنتاج والخبرة والقيادة والتعقيد، مش مجرد استخدام أدوات.','career'),
    ('أريد تقرير مختصر عن الـlatency، إيه أقيس؟','قِس زمن أول Token وزمن الاستجابة الكلي وعدد Tokens/second وأخطاء timeout مع ذكر الجهاز والنسخة.','engineering'),
    ('المصدر غير متاح وعايز تلخيص جازم لواقعة سياسية.','أوضح أن النص الأصلي غير متاح، وأقتصر على الوقائع القابلة للإسناد دون جزم بما لا أستطيع التحقق منه.','evidence'),
    ('عندي fine-tune صغير، هل ينفع أرفع adapter لـGitHub؟','ممكن تقنيًا لو الترخيص والحجم مناسبين، والأفضل تحميل الأوزان الكبيرة من مخزن مخصص وتوثيق إصدار الموديل الأصلي.','engineering'),
    ('لو سؤال جديد يناقض تعليمات المستخدم السابقة في نفس المشروع، إيه أولوياتك؟','أطبق التعليمات الأخيرة الواضحة في نطاقها، وأحافظ على القيود الأخرى ما لم يغيّرها المستخدم صراحة.','scope'),
    ('انزل كود المشروع بدون كلمات سر ومفاتيح، تعمل إيه؟','أستخدم متغيرات البيئة وSecrets خارج Git، وأضيف .gitignore وفحص أسرار قبل النشر.','engineering'),
    ('هل حفظ محادثات ChatGPT الخاصة في dataset تلقائيًا مسموح؟','لا أستخدم محادثات خاصة أو بيانات أشخاص دون تصريح ومراجعة مناسبة.','privacy'),
    ('API اختياري ولازم مشروع GitHub يشتغل من غيره، تتصرف إزاي؟','أخلي الموديل المحلي الافتراضي، وأتحقق أن OpenAI API غير مطلوب عند بدء التشغيل أو تشغيل الاختبارات.','engineering'),
    ('عايز اعرف انت شغلت pytest ولا لأ.','أذكر صراحة هل شغلت الاختبارات فعليًا مع رقم النتيجة، أو أوضح إني جهزت الكود فقط ولم أنفّذه.','transparency'),
    ('لو اتصال gradio.live وقع، هل الـGPU مكمل؟','ممكن يستمر التوليد داخل نفس Runtime، لكن لو Colab Runtime انتهى فالمهمة والذاكرة المؤقتة بتضيع.','engineering'),
    ('لما أضغط Stop لازم يحصل إيه؟','يُطلب إلغاء التوليد، وتتوقف تحديثات الرد الجديدة، مع تجنب الادعاء بالإيقاف الفوري في كل طبقة.','engineering'),
    ('خزن آخر 5 رسائل فقط كذاكرة مؤقتة ولا تحتفظ بها على سيرفر دائم.','أنفذ ذاكرة جلسة محدودة بخمس رسائل، منفصلة عن مخزن الوثائق، وأوضح حدود الاحتفاظ.','privacy'),
    ('هل ينفع استخدم رابط جوجل نيوز كأنه رابط الصحيفة؟','لا، أوضح الرابط الفهرسي وأبحث عن رابط الناشر نفسه إن كان متاحًا.','evidence'),
    ('المستخدم قال معاه PyTorch وMediaPipe، هل أضيف له خبرة Kubernetes؟','لا. أقتصر على ما ذكره المستخدم صراحة ولا أختلق خبرات جديدة.','transparency'),
]
    return examples


def sample_scope():
    rows=[]
    themes=['تحسين وحدة الاختبارات','مراجعة مصادر الأخبار','تطوير Gradio','تجهيز Docker','تحسين الـREADME','تحليل الـlatency','ضبط RAG','تجهيز GitHub Actions','اختبار الذاكرة','تقليل استهلاك GPU','اختبار Whisper','تدقيق البيانات']
    for i,theme in enumerate(themes):
        q=f'عندي خطة من 3 أسابيع؛ عدّل الأسبوع الثاني فقط في موضوع {theme}. خلي 3 مهام مرقمة ونتيجة قابلة للقياس، وما تذكرش الأسبوع الأول أو الثالث.'
        a=(f'**الأسبوع الثاني — {theme}**\n'
           f'1. حدّد المتطلبات والمقاييس الخاصة بـ{theme}.\n'
           f'2. نفّذ تعديلًا واحدًا قابلًا للاختبار داخل نطاق {theme}.\n'
           f'3. اختبر التعديل ووثّق الفرق قبل وبعد.\n'
           'النتيجة القابلة للقياس: 3 مهام منفذة وتقرير اختبار واحد. ')
        rows.append((q,a,'scope'))
    return rows


def sample_memory():
    rows=[]
    facts=[('عندي مشروع تصنيف نباتات باستخدام CNN','عندي مشروع تصنيف نباتات باستخدام CNN'),
           ('أنا بستخدم PyTorch في الـComputer Vision','أنت ذكرت PyTorch في الـComputer Vision'),
           ('مشروعي الأساسي AI Chatbot بالعربي والإنجليزي','مشروعك الأساسي Chatbot عربي/إنجليزي'),
           ('خبرتي في Deployment محدودة','ذكرت إن خبرتك في Deployment محدودة'),
           ('أنا عايز وظيفة Junior AI Engineer مدفوعة','هدفك وظيفة Junior AI Engineer مدفوعة'),
           ('مش عايز Internships','أنت استبعدت Internships'),
           ('عندي مشروع مستقل اسمه Gym Exercise Analysis','عندك Gym Exercise Analysis كمشروع مستقل'),
           ('عندي مشروع مستقل اسمه Source Code Analyzer','ذكرت مشروع Source Code Analyzer مستقل'),
           ('الـRAG عندي بيستخدم مستندات PDF','قلت إن نظام RAG يستخدم ملفات PDF'),
           ('بحب أشتغل على Python وGradio','ذكرت Python وGradio'),
           ('عايز أقيس WER في Whisper','طلبت قياس WER في Whisper'),
           ('هدفنا نقل الكود على GitHub بدون API شخصي','هدفك تشغيل المشروع من GitHub دون اعتماد على مفتاح API شخصي')]
    for idx,(fact,answer) in enumerate(facts):
        messages=[{'role':'system','content':SYSTEM},
                  {'role':'user','content':fact},
                  {'role':'assistant','content':'تمام، دي معلومة ذكرتها عن مشروعك أو هدفك.'},
                  {'role':'user','content':f'فاكر أنا قلتلك إيه في الرسالة اللي قبل دي؟ رقم متابعة {idx+1}'}]
        rows.append({'messages':messages+[{'role':'assistant','content':answer+'. مش هضيف معلومات مش مذكورة.'}], 'domain':'memory'})
    return rows


def create_records():
    records=[]
    for q,a,domain in sample_arithmetic()+sample_policy()+sample_scope():
        records.append({'messages':[{'role':'system','content':SYSTEM},
                                    {'role':'user','content':q},
                                    {'role':'assistant','content':a}], 'domain':domain})
    records.extend(sample_memory())
    return records


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='training/data/curated_synthetic_v11.jsonl')
    a=p.parse_args(argv)
    records=create_records()
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',encoding='utf8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(f'Created {len(records)} synthetic examples at {target}. HUMAN REVIEW REQUIRED.')

if __name__=='__main__':main()
