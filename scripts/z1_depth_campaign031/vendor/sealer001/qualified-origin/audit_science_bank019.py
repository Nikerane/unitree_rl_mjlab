"""019 file-only reader: retain a genuine early native timeout and full801 shadow.

The pinned014 reader keeps its closed-target, identity, provenance and file guards.
Exactly two counted native-assumption seams are derived from pinned e1/15e ASTs.
No physical source, endpoint/classifier arithmetic, or regular native path changes.
"""
from pathlib import Path
import ast,copy,hashlib,json,types

HERE=Path(__file__).resolve().parent
BASE_SHA='d7030182bc2e84b0c9a9f5be986c495bf815eb01bae6d3cfc01a4e7a775fa2dc'
body=(HERE/'fixture014/audit_science_bank014.py').read_bytes()
if hashlib.sha256(body).hexdigest()!=BASE_SHA:raise ValueError('frozen014 helper source differs')
base=types.ModuleType('science019_pinned014');base.__file__=str(HERE/'fixture014/audit_science_bank014.py')
exec(compile(body,base.__file__,'exec'),base.__dict__)
# Re-export the unchanged014 closed-target helpers for the reused fixture suite.
for name,value in vars(base).copy().items():
 if not name.startswith('_'):globals()[name]=value
DT=.00025
NATIVE_TIMEOUT_SAMPLE=80000


def native_censor_coverage(native,onset,ticks):
 """Early alternative only: onset-inclusive R; actual admitted 20s timeout."""
 r=native.get('contact_samples')
 if type(r)is not int or not 1<=r<=ticks:return False
 n=native.get('observed_samples');snapshot=native.get('snapshot',{})
 need(type(n)is int and n==NATIVE_TIMEOUT_SAMPLE==onset+r-1,'early native timeout timestamp/count differs')
 need(canonical(native.get('termination_terms'))==canonical(dict(time_out=True,nail_driven=False)),'early native requires actual timeout-only terms')
 need(snapshot.get('termination_reasons')=='time_out'and 'native_success_sample'in snapshot and snapshot['native_success_sample']is None and snapshot.get('native_success_time_s')is None,'early native success/reason differs')
 need(snapshot.get('miss')is False and type(snapshot.get('observed_samples'))is int and snapshot['observed_samples']==n,'early native snapshot count/miss differs')
 need(type(snapshot.get('observed_duration_s'))in(int,float)and math.isclose(snapshot['observed_duration_s'],n*DT,rel_tol=0,abs_tol=1e-12),'early native duration differs')
 need(type(snapshot.get('time_to_contact_s'))in(int,float)and math.isclose(snapshot['time_to_contact_s'],onset*DT,rel_tol=0,abs_tol=1e-12),'early native onset time differs')
 return True


def native_common_metrics(w,trace,world,native,intent,dense_metrics,require,near):
 """The normal complete-common check is retained verbatim on its old branch."""
 ticks=round(intent['policy']['native_window_s']/DT);r=native['contact_samples'];snapshot=native['snapshot']
 early=type(r)is int and 1<=r<=ticks
 if not early:
  common=dense_metrics(w,trace,world,0,200)
  require(snapshot.get('common_window',{}).get('window_complete') is True and snapshot['common_window']['window_samples']==200 and snapshot['loads']['common_window']['observed_samples']==200,'native common50ms endpoint missing')
  return common
 # audit_raw has already proved the real native marker and N=A+R-1 in the
 # dense trace. The producer freezes its common min(R,200) prefix at that N.
 p=min(r,200);complete=p==200
 expected=dict(window_samples=p,window_complete=complete,observed_duration_s=p*DT,required_samples=200,censored=not complete,censor_reason=None if complete else'time_out')
 require(canonical({k:snapshot.get('common_window',{}).get(k)for k in expected})==canonical(expected),'early native common prefix metadata differs')
 loads=snapshot.get('loads',{}).get('common_window',{})
 require(type(loads.get('observed_samples'))is int and loads['observed_samples']==p,'early native common prefix load count differs')
 common=dense_metrics(w,trace,world,0,p)
 # Existing15e peak/count parity below still runs, over this actual prefix.
 for key,expected in [('utilization',[v/c for v,c in zip(common['enforced']['peak'],w.LOAD_CAPS)]),('exceeded',[v>c for v,c in zip(common['enforced']['peak'],w.LOAD_CAPS)]),('exceedance_duration_s',[v*DT for v in common['enforced']['exceedance_samples']])]:
  require(near(loads.get(key),expected),'early native common prefix load metadata differs')
 return common


class NativeSeams(ast.NodeTransformer):
 """Only these exact pinned expression/statements may change."""
 def __init__(self,kind):self.kind=kind;self.counts={}
 def hit(self,name):self.counts[name]=self.counts.get(name,0)+1
 def visit_FunctionDef(self,node):
  if node.name!=('audit_raw'if self.kind=='e1'else'audit_measurements'):return node
  return self.generic_visit(node)
 def visit_Compare(self,node):
  old=ast.dump(ast.parse('ticks+1<=native_count<=ticks+80',mode='eval').body,include_attributes=False)
  if self.kind=='e1'and ast.dump(node,include_attributes=False)==old:
   self.hit('native_coverage_alternative')
   return ast.copy_location(ast.parse('(ticks+1<=native_count<=ticks+80) or native_censor_coverage(native,onset,ticks)',mode='eval').body,node)
  return self.generic_visit(node)
 def visit_Assign(self,node):
  old=ast.parse('common=dense_metrics(w,trace,i,0,200)').body[0]
  if self.kind=='own'and ast.dump(node,include_attributes=False)==ast.dump(old,include_attributes=False):
   self.hit('native_common_prefix_assignment')
   return ast.copy_location(ast.parse('common=native_common_metrics(w,trace,i,native,intent,dense_metrics,require,near)').body[0],node)
  return self.generic_visit(node)
 def visit_Expr(self,node):
  if self.kind=='own'and isinstance(node.value,ast.Call)and len(node.value.args)>1 and isinstance(node.value.args[1],ast.Constant)and node.value.args[1].value=='native common50ms endpoint missing':
   self.hit('native_common_check_moved_to_regular_branch');return None
  return self.generic_visit(node)


def dependencies(sealer_path,own_path):
 w,ebody=source(sealer_path,SEALER_SHA,'science019_e1');o,obody=source(own_path,OWN_SHA,'science019_15e')
 counts={}
 for body,name,kind,module in [(ebody,'audit_raw','e1',w),(obody,'audit_measurements','own',o)]:
  transform=NativeSeams(kind);tree=transform.visit(ast.parse(body));counts.update(transform.counts)
  adapted=ast.unparse(ast.fix_missing_locations(tree))
  module.__dict__.update(native_censor_coverage=native_censor_coverage,native_common_metrics=native_common_metrics)
  function,oldcounts=base.derive(adapted,name,kind,module.__dict__)
  if kind=='e1':w.audit_raw=function;ecounts=oldcounts
  else:w.audit_case_measurements=function;ocounts=oldcounts
 need(counts==dict(native_coverage_alternative=1,native_common_prefix_assignment=1,native_common_check_moved_to_regular_branch=1),'019 native seam structure changed')
 expected=dict(e1_case_literal=2,e1_reports_assignment=1,own_case_literal=1,own_audit_call=1,own_route_compare=1)
 need(ecounts|ocounts==expected,'closed derivative structure changed')
 w.adaptation_counts=ecounts|ocounts;w.native_adaptation_counts=counts;w.metadata=o.metadata;w.validate_terminal=o.validate_terminal
 return w


_original_measurements=base.audit_measurements

def audit_measurements(w,intent,root):
 result=_original_measurements(w,intent,root)
 result['science_reader_provenance']=dict(schema='z1_science_reader019_provenance_v1',producer=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),base014_helper_sha256=BASE_SHA,base_physical_source_pins=copy.deepcopy(result['source_pins']),native_adaptation_counts=w.native_adaptation_counts)
 # Added annotations keep native-horizon censoring distinct from common50.
 for case in result['cases']:
  for row in case['worlds']:
   ticks=round(intent['policy']['native_window_s']/DT);r=row['native_contact_samples']
   if row['outcome']=='CONTACT_COMPLETE'and type(r)is int and 1<=r<=ticks:
    row['native_censor019']=dict(native_horizon_censored=True,native_horizon_required_ticks=ticks,native_contact_observed_points=r,common50_censored=r<200,common50_observed_points=min(r,200),reason='time_out',fixed801_shadow_complete=True)
 return result

# The original014 audit function resolves these two names in its own globals;
# all remaining guard code and its argument/CLI contract are unchanged bytes.
base.dependencies=dependencies;base.audit_measurements=audit_measurements

def audit(sealer_path,own_path,slot_index,supervisor_path=SUPERVISOR_PATH):return base.audit(sealer_path,own_path,slot_index,supervisor_path)
if __name__=='__main__':base.main()
