#!/usr/bin/env python3
"""
Script di compilazione per progetto_strutturale.tex utilizzando la libreria pdflatex.
Risolve le incompatibilità native di MiKTeX su Windows (flag -interaction e directory Temp).
Include sia la modalità di esecuzione singola che la modalità in ascolto (--watch).
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from subprocess import PIPE
from pdflatex import PDFLaTeX

class MiKTeXPDFLaTeX(PDFLaTeX):
    """
    Sottoclasse personalizzata di PDFLaTeX per risolvere due problemi noti su Windows/MiKTeX:
    1. Corregge il flag errato '-interaction-mode' in '-interaction'.
    2. Utilizza una cartella temporanea locale anziché la cartella Temp di sistema di Windows,
       evitando conflitti con la cache e i log di MiKTeX/VCRedist presenti in AppData/Local/Temp.
    """
    def get_run_args(self, tex_filename=None):
        # 1. Correggi il flag di interazione per MiKTeX
        if '-interaction-mode' in self.params:
            mode = self.params.pop('-interaction-mode')
            self.params['-interaction'] = mode
        
        args = [k+('='+v if v is not None else '') for k, v in self.params.items()]
        args.insert(0, 'pdflatex')
        
        # MiKTeX richiede il percorso fisico del file anziché lo stream stdin
        if tex_filename:
            args.append(tex_filename)
        return args

    def create_pdf(self, keep_pdf_file: bool = False, keep_log_file: bool = False, env: dict = None):
        if self.interaction_mode is not None:
            self.add_args({'-interaction-mode': self.interaction_mode})
        
        out_dir = self.params.get('-output-directory')
        filename = self.params.get('-jobname')
        
        if filename is None:
            filename = self.job_name
        if out_dir is None:
            out_dir = ""
        
        # 2. Utilizza una cartella temporanea locale per evitare i conflitti di MiKTeX in Windows Temp
        local_temp_dir = os.path.abspath('.pdflatex_temp_' + filename)
        os.makedirs(local_temp_dir, exist_ok=True)
        
        try:
            self.set_output_directory(local_temp_dir)
            self.set_jobname(filename)
            
            tex_path = os.path.join(local_temp_dir, filename + '.tex')
            if isinstance(self.latex, str):
                with open(tex_path, 'w', encoding='utf-8') as f:
                    f.write(self.latex)
            else:
                with open(tex_path, 'wb') as f:
                    f.write(self.latex)
    
            args = self.get_run_args(tex_path)
            fp = subprocess.run(args, env=env, timeout=60, stdout=PIPE, stderr=PIPE)
            
            pdf_path = os.path.join(local_temp_dir, filename + '.pdf')
            log_path = os.path.join(local_temp_dir, filename + '.log')
            
            if os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as f:
                    self.pdf = f.read()
                if keep_pdf_file:
                    dest_pdf = os.path.join(out_dir, filename + '.pdf')
                    shutil.copyfile(pdf_path, dest_pdf)
            else:
                self.pdf = None
                
            if os.path.exists(log_path):
                with open(log_path, 'rb') as f:
                    self.log = f.read()
                if keep_log_file:
                    dest_log = os.path.join(out_dir, filename + '.log')
                    shutil.copyfile(log_path, dest_log)
            else:
                self.log = None
                
        finally:
            # Ripulisci la cartella temporanea locale
            if os.path.exists(local_temp_dir):
                try:
                    shutil.rmtree(local_temp_dir)
                except Exception:
                    pass
        
        return self.pdf, self.log, fp

def compile_document(tex_file):
    print(f"[*] Avvio compilazione di '{tex_file}' con pdflatex...")
    start_time = time.time()
    
    if not os.path.exists(tex_file):
        print(f"[!] Errore: Il file '{tex_file}' non esiste.")
        return False

    try:
        pdfl = MiKTeXPDFLaTeX.from_texfile(tex_file)
        pdf, log, cp = pdfl.create_pdf(keep_pdf_file=True)
        
        elapsed = time.time() - start_time
        if cp.returncode == 0 and pdf is not None:
            pdf_name = os.path.splitext(tex_file)[0] + '.pdf'
            print(f"[+] Compilazione completata con successo in {elapsed:.2f}s!")
            print(f"[+] File generato/sovrascritto: {pdf_name}")
            return True
        else:
            print(f"[!] Errore durante la compilazione (Codice di uscita: {cp.returncode}) in {elapsed:.2f}s.")
            if cp.stdout:
                print("--- STDOUT ---")
                print(cp.stdout.decode('latin1', errors='replace'))
            if cp.stderr:
                print("--- STDERR ---")
                print(cp.stderr.decode('latin1', errors='replace'))
            return False
            
    except Exception as e:
        print(f"[!] Eccezione durante la compilazione: {e}")
        return False

def watch_document(tex_file):
    print(f"[*] Modalità in ascolto (--watch) attiva su '{tex_file}'.")
    print("[*] Premi Ctrl+C per interrompere.\n")
    
    if not os.path.exists(tex_file):
        print(f"[!] Errore: Il file '{tex_file}' non esiste.")
        return

    last_mtime = os.path.getmtime(tex_file)
    compile_document(tex_file)
    
    try:
        while True:
            time.sleep(1)
            try:
                current_mtime = os.path.getmtime(tex_file)
                if current_mtime != last_mtime:
                    print(f"\n[*] Rilevata modifica in '{tex_file}'. Ricompilazione in corso...")
                    last_mtime = current_mtime
                    compile_document(tex_file)
            except FileNotFoundError:
                pass
    except KeyboardInterrupt:
        print("\n[*] Modalità in ascolto interrotta dall'utente.")

def main():
    parser = argparse.ArgumentParser(description="Compila un documento LaTeX in PDF usando la libreria pdflatex.")
    parser.add_argument("tex_file", nargs="?", default="nome_file.tex", help="Il file .tex da compilare (default: noem_file.tex)")
    parser.add_argument("--watch", action="store_true", help="Rimani in ascolto per modifiche al file e ricompila automaticamente")
    
    args = parser.parse_args()
    
    if args.watch:
        watch_document(args.tex_file)
    else:
        success = compile_document(args.tex_file)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
