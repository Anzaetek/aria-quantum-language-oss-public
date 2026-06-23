" Aria Quantum Language filetype detection.
" Every .aria file in this project is Aria Quantum.
augroup AriaQuantumFiletypeDetect
  autocmd!
  autocmd BufRead,BufNewFile *.aria setlocal filetype=aria_quantum
augroup END
