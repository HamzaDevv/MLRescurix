"""Generate the committed v1 schema and shared adversarial fixture corpus."""
import copy
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
I = {"type":"string","minLength":1,"maxLength":128,"pattern":"[A-Za-z0-9_.:-]+"}
T = {"type":"string","minLength":1,"maxUtf8Bytes":4096}
N = {"type":"integer","minimum":1,"maximum":9007199254740991}
C = {"type":"string","pattern":"0|[1-9][0-9]{0,18}"}
L = {"type":"array","items":I,"maxItems":64}
def obj(fields):
    return {"type":"object","properties":fields,"required":list(fields),"additionalProperties":False}
def nullable(schema):
    return {"oneOf":[schema,{"type":"null"}]}
context=obj({"role_id":I,"role_version":I,"prompt_hash":I,"model":nullable(T),"phase":T,"hypothesis":nullable(T),"base_hash":I,"code_hash":I,"evidence":L,"artifacts":L,"pending_questions":L,"calls":L,"jobs":L,"next_action":T,"stop_conditions":L})
checkpoint=obj({"checkpoint_id":I,"sequence":N,"project_id":I,"branch_id":I,"task_id":I,"spec_revision":N,"intent_revision":N,"generation":N,"event_cursor":C,"boundary":T,"context":{"$ref":"#/$defs/context"}})
def frame(kind,payload):
    return obj({"version":{"const":1},"kind":{"const":kind},"payload":payload})
schema={"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"urn:mlrescurix:frame:1","$defs":{"context":context,"checkpoint":checkpoint},"oneOf":[frame("hello",obj({"protocol":{"const":"mlrescurix/1"},"worker_id":I})),frame("observation",obj({"project_id":I,"task_id":I,"action_id":I,"generation":N,"phase":T,"detail":T})),frame("checkpoint",{"$ref":"#/$defs/checkpoint"})]}
serialized=json.dumps(schema,indent=2)+"\n"
(ROOT/'contracts/v1/frame.schema.json').write_text(serialized)
(ROOT/'python/src/mlrescurix_worker/protocol/frame.schema.json').write_text(serialized)
ctx={"role_id":"researcher","role_version":"1","prompt_hash":"prompt-sha256","model":None,"phase":"foundation","hypothesis":None,"base_hash":"base-sha256","code_hash":"code-sha256","evidence":[],"artifacts":[],"pending_questions":[],"calls":[],"jobs":[],"next_action":"inspect committed evidence","stop_conditions":["budget_exhausted"]}
hello={"version":1,"kind":"hello","payload":{"protocol":"mlrescurix/1","worker_id":"fake-worker"}}
obs={"version":1,"kind":"observation","payload":{"project_id":"p0","task_id":"t0","action_id":"a0","generation":2,"phase":"scripted","detail":"synthetic observation, not official evidence"}}
cp={"version":1,"kind":"checkpoint","payload":{"checkpoint_id":"t0.1","sequence":1,"project_id":"p0","branch_id":"b0","task_id":"t0","spec_revision":1,"intent_revision":1,"generation":2,"event_cursor":"12","boundary":"yield","context":ctx}}
fixtures=[]
def put(name,value,accepted):
    path=f'{name}.json';(ROOT/'contracts/v1/fixtures'/path).write_text(json.dumps(value,ensure_ascii=False)+"\n");fixtures.append({"file":path,"accepted":accepted})
for name,value in [('hello',hello),('observation',obs),('checkpoint',cp)]:put(name,value,True)
for name,base,route,value in [
    ('version',hello,['version'],2),('float_version',hello,['version'],1.0),
    ('protocol',hello,['payload','protocol'],'mlrescurix/2'),('worker_path',hello,['payload','worker_id'],'../legacy'),
    ('kind',obs,['kind'],'official_score'),('zero_generation',obs,['payload','generation'],0),
    ('float_generation',obs,['payload','generation'],2.0),('huge_generation',obs,['payload','generation'],9007199254740992),
    ('bool_generation',obs,['payload','generation'],True),('negative_generation',obs,['payload','generation'],-1),
    ('empty_text',obs,['payload','detail'],''),('large_utf8',obs,['payload','detail'],'é'*3000),
    ('numeric_cursor',cp,['payload','event_cursor'],12),('leading_zero_cursor',cp,['payload','event_cursor'],'012'),
    ('zero_revision',cp,['payload','spec_revision'],0),('unknown_role_field',cp,['payload','context','privileged'],True),
    ('fake_score',obs,['payload','official_score'],0.99),('fake_finalize',obs,['payload','final_model'],'forged'),
    ('unknown_envelope',hello,['extra'],1),('bad_next_action',cp,['payload','context','next_action'],''),
]:
    changed=copy.deepcopy(base);node=changed
    for key in route[:-1]:node=node[key]
    node[route[-1]]=value;put(name,changed,False)
missing=copy.deepcopy(cp);del missing['payload']['context']['calls'];put('missing_calls',missing,False)
(ROOT/'contracts/v1/fixtures/manifest.json').write_text(json.dumps(fixtures,indent=2)+"\n")
