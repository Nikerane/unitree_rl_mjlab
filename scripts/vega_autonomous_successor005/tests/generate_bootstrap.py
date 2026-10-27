"""Generate/check the isolated stdlib security bootstrap; never import helpers.

Canonical maintained source is resident_archive.BOOTSTRAP, read using AST only.
This development utility lives outside the production source closure.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path

BEGIN='# BEGIN GENERATED SECURITY BOOTSTRAP — edit archive/resident_archive.py BOOTSTRAP\n'
END='# END GENERATED SECURITY BOOTSTRAP\n'

def generated_entry(resident_source,entry_source):
    tree=ast.parse(resident_source)
    values=[node.value for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='BOOTSTRAP' for target in node.targets)]
    if len(values)!=1:raise ValueError('one canonical BOOTSTRAP literal required')
    bootstrap=ast.literal_eval(values[0])
    if not isinstance(bootstrap,str):raise ValueError('canonical bootstrap must be source text')
    if entry_source.count(BEGIN)!=1 or entry_source.count(END)!=1:raise ValueError('exact generated bootstrap markers required')
    start=entry_source.index(BEGIN)+len(BEGIN);stop=entry_source.index(END,start)
    return entry_source[:start]+bootstrap.rstrip()+'\n'+entry_source[stop:]

def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--check',action='store_true');group.add_argument('--write',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];path=root/'independent_verify.py';existing=path.read_text();expected=generated_entry((root/'archive/resident_archive.py').read_text(),existing)
    if args.check:
        if expected!=existing:raise SystemExit('isolated bootstrap drift; run tests/generate_bootstrap.py --write')
        print('isolated security bootstrap matches canonical source')
    else:path.write_text(expected)

if __name__=='__main__':main()
