import json, random
random.seed(123)

STYLE_BODIES = {
 "formal":"We are pleased to announce an opening for a {role} at {company}. The successful candidate will contribute to our team and uphold professional standards. This is a {mode} position based in {city}.",
 "casual":"hey! so we are hiring a {role} at {company} pretty chill role, {mode}, based in {city}. lmk if ur interested!",
 "broken":"We are having job for {role} in our company {company}. Is {mode} work in {city}. You are do good work for us and we are happy.",
 "corporate":"{company} is seeking a qualified {role} to join our organization. This {mode} role is located in {city} and offers involvement in key operational activities.",
 "enthusiastic":"AMAZING opportunity!! {company} needs a {role} RIGHT NOW!! {mode} role in {city} - this is your chance to shine!!",
 "terse":"{role} wanted. {company}. {mode}. {city}.",
}
SIGNALS = {
 "fee_demand":{"formal":"To finalize your appointment, a one-time registration fee of {amt} is required for administrative processing.","casual":"oh just a small registration fee of {amt} to get u started, then ur all set!","broken":"You must pay fee {amt} for confirm the job position before start.","corporate":"Candidates are required to remit a processing fee of {amt} prior to onboarding.","enthusiastic":"Just {amt} registration fee and YOU ARE IN!! Do not miss out!!","terse":"Registration fee: {amt}. Required."},
 "unrealistic_income":{"formal":"Compensation may reach {amt} per week for candidates who perform consistently.","casual":"u can literally make {amt} a week just working from home lol","broken":"Income is very much, you earn {amt} in week easily no problem.","corporate":"Earning potential extends to {amt} weekly, subject to performance metrics.","enthusiastic":"Earn up to {amt} PER WEEK!! Unlimited income potential!!","terse":"Income: up to {amt}/week."},
 "external_contact":{"formal":"Please direct your application to our hiring manager via WhatsApp at {phone}.","casual":"just msg us on whatsapp {phone} or telegram, easier that way!","broken":"You send your details in WhatsApp number {phone} for we contact you.","corporate":"Interested applicants should make contact through WhatsApp: {phone}.","enthusiastic":"Message us on Telegram NOW at {phone}!! Fast response!!","terse":"Contact WhatsApp {phone}."},
 "urgency":{"formal":"We kindly request that you confirm within 24 hours, as positions are limited.","casual":"hurry tho, spots r filling up fast!! reply asap","broken":"Is urgent, you must reply now now, limited seat only.","corporate":"Immediate response is advised; available positions are limited.","enthusiastic":"ACT NOW!! Only a few spots left!! Apply IMMEDIATELY!!","terse":"Urgent. Limited slots. Reply now."},
 "identity_request":{"formal":"Kindly provide a copy of your passport and bank details to complete verification.","casual":"oh can u send a pic of ur passport n bank info? just for verification","broken":"You give passport copy and bank account for we make your account.","corporate":"Please submit passport identification and banking information for processing.","enthusiastic":"Just send your passport and bank details and you are READY TO EARN!!","terse":"Send passport + bank details."},
 "low_requirements":{"formal":"No prior experience is necessary and no interview is required.","casual":"no experience needed at all, anyone can do this!","broken":"Not need experience, not need interview, everybody can apply easy.","corporate":"This role requires no previous experience and involves no formal interview.","enthusiastic":"NO experience?? NO problem!! ANYONE can apply!!","terse":"No experience. No interview. Anyone."},
}
GENUINE_ADD = {"formal":"The role offers a competitive salary, medical benefits, and clear career progression. Apply through our official careers portal.","casual":"decent pay, health benefits, good team vibes apply on our careers page!","broken":"We give good salary and benefit. You apply in our website careers page please.","corporate":"This position includes a structured compensation package, benefits, and defined responsibilities. Applications via the official portal.","enthusiastic":"GREAT salary!! AMAZING benefits!! Apply on our careers page today!!","terse":"Competitive pay. Benefits. Apply via careers portal."}

ROLES=["Data Entry Clerk","Customer Service Rep","Marketing Assistant","Online Chat Agent","Sales Executive","Administrative Assistant","Virtual Assistant","Content Reviewer"]
COMPANIES=["Nexa Solutions","BrightPath Ltd","Global Connect","Prime Works","Metro Services","Apex Group","Vertex Inc","Summit Partners"]
CITIES=["Kuala Lumpur","Singapore","Remote","Bangkok","Jakarta","Manila","Hybrid - KL"]
MODES=["remote","full-time","part-time","work-from-home","hybrid"]
AMOUNTS=["RM150","RM200","USD 200","RM500","USD 1500","RM8000","USD 3000","RM300"]
PHONES=["+60 12-345 6789","+1 555-0192","+60 16-371 7555","@hiring_now","+63 917 555 0123"]

def fill(t):
    return (t.replace("{role}",random.choice(ROLES)).replace("{company}",random.choice(COMPANIES)).replace("{city}",random.choice(CITIES)).replace("{mode}",random.choice(MODES)).replace("{amt}",random.choice(AMOUNTS)).replace("{phone}",random.choice(PHONES)))

out=[]
styles=list(STYLE_BODIES.keys())
N=40
for style in styles:
    for _ in range(N):
        body=fill(STYLE_BODIES[style])
        chosen=random.sample(list(SIGNALS.keys()),random.randint(1,3))
        parts=[body]+[fill(SIGNALS[s][style]) for s in chosen]
        out.append({"text":" ".join(parts),"label":"fraudulent","source":"style_varied","style":style,"signals":chosen,"synthetic":True})
for style in styles:
    for _ in range(N):
        body=fill(STYLE_BODIES[style])
        out.append({"text":body+" "+GENUINE_ADD[style],"label":"genuine","source":"style_varied","style":style,"synthetic":True})
random.shuffle(out)
with open("style_varied_train.jsonl","w",encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r)+"\n")
print("Generated",len(out),"examples. Written to style_varied_train.jsonl")
