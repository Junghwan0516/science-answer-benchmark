"""Rebuild data-analysis stages without fitting models or changing split rules."""
from pathlib import Path
import argparse,contextlib,hashlib,io,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['advanced','full'],default='advanced',help='Full also rebuilds original audit and existing versioned splits.')
    parser.add_argument('--refresh-checksums',action='store_true',help='Update the release checksum file only after checks succeed; review resulting changes before committing.')
    args=parser.parse_args()
    if args.stage=='full':
        from build_analysis import build as audit
        from build_splits import build as split
        audit();split()
    from analysis_extensions import build as analyze
    from build_warehouse import build as warehouse
    from prepare_extension_notebooks import prepare
    analyze();warehouse();prepare()
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],check=True,cwd=ROOT)
    # Hash only tracked or explicitly generated portable artifacts; the local SQLite DB is ignored.
    if args.refresh_checksums:
        names=sorted(set(subprocess.check_output(['git','ls-files','-co','--exclude-standard'],cwd=ROOT,text=True).splitlines()))
        names=[n for n in names if n!='checksums.sha256' and (ROOT/n).is_file()]
        (ROOT/'checksums.sha256').write_text('\n'.join(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()+'  '+n for n in names)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_release.py')],check=True,cwd=ROOT)
    print('Pipeline complete. No prediction model was trained.')

if __name__=='__main__':main()
