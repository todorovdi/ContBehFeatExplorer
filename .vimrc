let g:pymode = 1
let g:pymode_warnings = 1
let g:pymode_paths = []
let g:pymode_trim_whitespaces = 1
let g:pymode_options = 1
let g:pymode_folding = 0
let g:pymode_indent = 1
let g:pymode_indent_hanging_width = &shiftwidth
let g:pymode_indent_hanging_width = 4
let g:pymode_motion = 1

let g:pymode_doc = 1
let g:pymode_doc_bind = 'K'
" automatic
let g:pymode_virtualenv = 0
let g:pymode_virtualenv_path = $VIRTUAL_ENV

let g:pymode_run = 0

let g:pymode_rope = 1
let g:pymode_rope_refix = '<C-c>'

augroup unset_folding_in_insert_mode
    autocmd!
    autocmd InsertEnter *.py setlocal foldmethod=marker
    autocmd InsertLeave *.py setlocal foldmethod=expr
augroup END

" e, new, vnew
let g:pymode_rope_goto_definition_cmd = 'e'

let g:pymode_rope_rename_bind = '<C-c>rr'
" current module (not that git will miss it)
let g:pymode_rope_rename_module_bind = '<C-c>r1r'

"Organize imports sorts imports, too. It does that according to PEP8. Unused
"imports will be dropped.
let g:pymode_rope_organize_imports_bind = '<C-c>ro'

"Insert import for current word under cursor     *'g:pymode_rope_autoimport_bind'*
"Should be enabled |'g:pymode_rope_autoimport'|
">
let g:pymode_rope_autoimport_bind = '<C-c>ra'

let g:pymode_rope_extract_method_bind = '<C-c>rm'
let g:pymode_rope_extract_variable_bind = '<C-c>rl'
let g:pymode_rope_use_function_bind = '<C-c>ru'
let g:pymode_rope_change_signature_bind = '<C-c>rs'

let g:pymode_rope_lookup_project = 1
let g:pymode_rope_complete_on_dot = 0

let g:pymode_lint_ignore = ["E201", "E221", "E262", "E303", "E202", "E251","E211", "E265", "E203", "E222", "E225", "E302", "E122", "E231", "E501", "W","E702", "E261", "C901", "E402", "E127", "E128", "E271", "E266", "E703", "E116", "E272", "E401", "E502", "E731", "E114", "E115", "E124", "E125"]

":PymodeRopeUndo* -- Undo last changes in the project
":PymodeRopeRedo* -- Redo last changes in the project

set foldmethod=indent


"  By default you can use <Ctrl-Space> for autocompletion. The first entry will
"  be automatically selected and you can press <Return> to insert the entry in
"  your code. <C-X><C-O> and <C-P>/<C-N> works too.
" 
"  Autocompletion is also called by typing a period in |Insert| mode by default.
" 
" By default when you press *<C-C>g* on any object in your code you will be moved
" to definition.


""""
" [[    Jump to previous class or function (normal, visual, operator modes)
" ]]    Jump to next class or function  (normal, visual, operator modes)
" [M    Jump to previous class or method (normal, visual, operator modes)
" ]M    Jump to next class or method (normal, visual, operator modes)
" aC    Select a class. Ex: vaC, daC, yaC, caC (operator modes)
" iC    Select inner class. Ex: viC, diC, yiC, ciC (operator modes)
" aM    Select a function or method. Ex: vaM, daM, yaM, caM (operator modes)
" iM    Select inner function or method. Ex: viM, diM, yiM, ciM (operator modes)
" V     Select logical line. Ex: dV, yV, cV (operator modes), also works with count
