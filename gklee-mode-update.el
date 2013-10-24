(require 'cl)

;(defvar gklee-home-path "/home/sawaya/gklee/")
(defvar gklee-home-path nil)
(defvar gklee-compile-bin "bin/klee-l++")
(defvar gklee-compile-buffer-name "*gklee-compile-debug*")
(defvar gklee-compile-args '("-g"))
(defvar gklee-opt-level nil)
(defcustom gklee-user-compile-args '() "Command line arguments to be passed to klee-l++ -- should be entered in list form, i.e. '(\"arg1\" \"arg2\"" :type '(repeat string))
(defvar gklee-comp-out-path nil)
(defvar gklee-default-path "./")
(defvar gklee-run-buffer-name "*gklee-run*")
(defvar gklee-run-debug-buffer-name "*gklee-run-debug*")
(defvar gklee-run-user-args '())
(defvar gklee-user-program-args nil)
(defvar gklee-info-notice "GKLEE, copyright (c) 2011, Gauss Group, University of Utah\n")
(defvar gklee-trace-buffer nil)
(defvar gklee-active-filter-buffer "*gklee-active-filters*")
(defvar gklee-available-filter-buffer "*gklee-available-filters*")
(defvar gklee-source-path nil)
(defvar gklee-filter-obarray)
(defvar gklee-current-trace-line 1)
(defconst gklee-filter-obarray-size 8191 "This is the size to initialize the
gklee-filter-obarray object to -- it must be a vector, and, according to elisp manual (23.3) performs best with a size prime or power of 2 - 1.  Was chosen for max blocksize.x + max gridsize.x (of dev cap 2.0 CUDA)")
(defvar gklee-warp-count 0)
;;(defvar gklee-trace-first-err-line nil)
;;(defvar gklee-trace-error-lines 0)
;; This is being replaced by gklee-error-alist
;; (defvar gklee-error-htable (make-hash-table :test 'equal)
;;   "This is where errors are stored for lookup in order to match to the other threads
;; involved in the error -- in form:
;; [key] = location [value] = (([errorType] [location] [optional location])([errorType] [location] ...))location = trace#:file:lineno:block#:thread")
(defvar gklee-error-alist nil)
(defvar gklee-trace-error-list nil
  "This is a circular list, such that the cdr of the last element points to the first element --
it is a list of (point)s in the trace buffer that are potential error locations.  They
are only potential because we don't maintain enough state to be certain that the execution pass
at the indicated location is the one that caused the error -- we only maintain blk:thd:file:line")
(defconst gklee-threads-per-warp 32)
(defvar gklee-block-syms nil)
(defvar gklee-thread-syms nil)
(defvar gklee-file-syms nil)
(defvar gklee-loc-syms nil)
(defvar gklee-warp-syms nil)

(setq gklee-options-alist
      '(("--ignore-concur-bug" . 0)
	("--check-BC" . 1)
	("--check-MC" . 1)
	("--check-WD" . 1)
					;("--check-Race" . 1)
	("--check-volatile" . 1)
	("--check-barrier-redundant" . 0)
	("--device-capability" . 2)
	("--reduce-redundant-tests" . 0)
	("--bc-cov" . 0)
	("--reduce-path" . "")
	("--verbose" . 0)
	("--check-level" . 1)
	)
      )

;;GLOBAL KEYMAPS
(global-set-key "\M-gtcb" 'gklee-toggle-concurrency-bug)
(global-set-key "\M-gtbc" 'gklee-toggle-bank-conflicts)
(global-set-key "\M-gtmc" 'gklee-toggle-memory-coalesce)
(global-set-key "\M-gtwd" 'gklee-toggle-warp-divergence)
					;(global-set-key "\M-gtrc" 'gklee-toggle-race-condition)
(global-set-key "\M-gor" 'gklee-open-remote-package)
(global-set-key "\M-gtcv" 'gklee-toggle-check-volatle)
(global-set-key "\M-gtrb" 'gklee-toggle-check-barrier)
(global-set-key "\M-gsdc" 'gklee-set-device-capability)
(global-set-key "\M-gtrt" 'gklee-toggle-reduce-tests)
(global-set-key "\M-gtgc" 'gklee-toggle-bytecode-coverage)
(global-set-key "\M-gspr" 'gklee-set-path-reduction)
(global-set-key "\M-gtv" 'gklee-toggle-verbose)
(global-set-key "\M-gscl" 'gklee-set-check-level)
(global-set-key "\M-gr" 'gklee-run)
(global-set-key "\M-gupa" 'gklee-modify-user-prog-args)
(global-set-key "\M-guca" 'gklee-modify-user-compile-args)
(global-set-key "\M-gaga" 'gklee-modify-addn-gklee-args)
(global-set-key "\M-gk" 'gklee-kill)

(defun gklee-open-remote-package()
  (interactive)
  (let((package (read-file-name "Enter location of output.tgz:\n"))
       (curr-dir (make-temp-file "gklee" t))
       (newname (make-temp-file "gklee")))
    (gklee-create-windows 
    (copy-file package newname t)
    ;;Executes tar in a bash shell in order to use shell filesystem abrevs, like ~ and *
    (call-process  "/bin/bash" nil gklee-run-debug-buffer-name
		  nil "-c" (concat "/bin/tar -C " curr-dir " -zxf " newname))
    (setq gklee-source-path
	  (concat (file-name-as-directory curr-dir)
		  (with-temp-buffer 
		    (insert-file-contents (concat (file-name-as-directory curr-dir) "lastran"))
		    (buffer-string))))
    (setq buffer (get-buffer-create gklee-run-buffer-name))
    (setq buffer-debug (get-buffer-create gklee-run-debug-buffer-name))
    (with-current-buffer (setq gklee-source-buffer  (find-file gklee-source-path)))
    (with-current-buffer buffer
      (use-local-map
       (let ((map (make-sparse-keymap)))
	 (define-key map "k" 'gklee-kill)
	 map))
      (setq gklee-source-mark (point-marker)) ;;sets the buffer property of gklee-source-mark?  may not be necessary
      (gklee-reset-run-state)
      (erase-buffer)
      (buffer-disable-undo)
      (setq buffer-read-only t)
      (setq output (with-temp-buffer
		     (insert-file-contents (concat (file-name-as-directory curr-dir) "gklee_out"))
		     (buffer-string)))
      (gklee-process-output buffer output)
      (gklee-full-refresh)
      )
    ))

(defun gklee-circularize-list (lst)
  "this function must be called with a proper list (listp lst)"
  (let ((elm lst))
    (while (cdr elm)
      (setq elm (cdr elm)))
    (if lst
	(setcdr elm lst)
      )
    ))

(defun gklee-modify-addn-gklee-args()
  "User may enter additional arguments for GKLEE program"
  (interactive)
  (gklee-user-mod-list gklee-run-user-args "Enter args to pass to GKLEE: ")
  )

(defun gklee-modify-user-compile-args()
  "User may set options to be passed to LLVM compiler"
  (interactive)
  (gklee-user-mod-list gklee-user-compile-args "Enter args for klee-l++: ")
  )

(defun gklee-modify-user-prog-args ()
  "This allows user to enter command line args
for the program under test by GKLEE"
  (interactive)
  (gklee-user-mod-list gklee-user-program-args "Enter user program args: ")
		      
  )

(defun gklee-user-mod-list (lst prompt)
  "This allows users to modify a given list (such as command
line arguments"
  (interactive)
  (setq lst (split-string 
	     (read-string prompt 
			  (mapconcat 'identity 
				     lst " "))
	     " "))
  
  )

(defun gklee-toggle-run-option(option)
  (let ((old-value (cdr (assoc option gklee-options-alist))))
    (if (equal old-value 0)
	(progn
	  (message (concat "setting " option " to 1"))
	  (setcdr (assoc option gklee-options-alist) 1)
	  )
      (message (concat "setting " option " to 0"))
      (setcdr (assoc option gklee-options-alist) 0)
      )
    ))

(defun gklee-toggle-concurrency-bug()
  (interactive)
  (gklee-toggle-run-option "--ignore-concur-bug"))

(defun gklee-toggle-bank-conflicts()
  (interactive)
  (gklee-toggle-run-option "--check-BC"))

(defun gklee-toggle-memory-coalesce()
  (interactive)
  (gklee-toggle-run-option "--check-MC"))

(defun gklee-toggle-warp-divergence()
  (interactive)
  (gklee-toggle-run-option "--check-WD"))

;; (defun gklee-toggle-race-condition()
;;   (interactive)
;;   (gklee-toggle-run-option "--check-Race"))

(defun gklee-toggle-check-volatile()
  (interactive)
  (gklee-toggle-run-option "--check-volatile"))

(defun gklee-toggle-reduce-tests()
  (interactive)
  (gklee-toggle-run-option "--reduce-redundant-tests"))

(defun gklee-toggle-bytecode-coverage()
  (interactive)
  (gklee-toggle-run-option "--bc-cov"))

(defun gklee-toggle-verbose()
  (interactive)
  (gklee-toggle-run-option "--verbose"))

(defun gklee-toggle-check-barrier()
  (interactive)
  (gklee-toggle-run-option "--check-barrier-redundant"))

(defun gklee-set-device-capability(n)
  (interactive "nDevice Capability (0)1.0-1.1 (1)1.2-1.3 (2)2: ")
  (if (and (< n 3)(> n -1))
      (setcdr (assoc "--device-capability" gklee-options-alist) n)
    (message "set device capability to default of 2")
    (setcdr (assoc "--device-capability" gklee-options-alist) 2)
    ))

(defun gklee-set-path-reduction(BorT)
  (interactive "sset path reduction to b(line/branch) or t(line or branch): ")
  (if (or (equal "b" BorT)(equal "t" BorT)(equal "" BorT))
      (setcdr (assoc "--reduce-path" gklee-options-alist) BorT)
    (message "set path reduction to default: ''")
    (setcdr (assoc "--reduce-path" gklee-options-alist) "")
    ))

(defun gklee-set-check-level(n)
  (interactive "nset race check level (0)none (1)shared mem (2)full: ")
  (if (and (> n -1)(< n 3))
      (setcdr (assoc "--check-level" gklee-options-alist) n)
    (message "setting race check level to default of 1")
    (setcdr (assoc "--check-level" gklee-options-alist) 1)
    ))

(make-local-variable 'gklee-compile-process)
(make-local-variable 'gklee-run-process) 
(make-local-variable 'gklee-run-buffer)
(make-local-variable 'gklee-unparsed)

(put 'gklee-line-cat 'mouse-face 'highlight)

(put 'gklee-filter-cat 'mouse-face 'highlight)

(defvar gklee-unparsed "")
(defvar gklee-record nil)
(defvar gklee-race-info-list nil)
(defvar gklee-deadlock-info-list nil)
(defvar gklee-assertion-info-list nil)
(defvar gklee-config-info-list nil)
(defvar gklee-bankcon-info-list nil)
(defvar gklee-memcol-info-list nil)
(defvar gklee-warpdiv-info-list nil)
(defvar gklee-summary-info-list nil)
(defvar gklee-process-record-count 0)

(defvar gklee-trace-source-buffers nil)

(defvar gklee-trace-instr-array nil)
(defvar gklee-asm-visible t)

(setq gklee-filter-keymap
      (let ((map (make-sparse-keymap)))
	(define-key map [mouse-1] 'gklee-filter-event)
	map))

(setq gklee-trace-keymap
      (let ((map (make-sparse-keymap)))
	(define-key map "\d" 'gklee-exit-trace)
	(define-key map [mouse-1] 'gklee-goto-line-event)
	(define-key map "\^M" 'gklee-goto-line-key)
	(define-key map "n" 'gklee-next-trace-inst)
	(define-key map "p" 'gklee-prev-trace-inst)
	(define-key map "\M-gst" 'gklee-show-thread)
	(define-key map "\M-gsw" 'gklee-show-warp)
	(define-key map "\M-gsf" 'gklee-show-file)
	(define-key map "\M-gsb" 'gklee-show-block)
	(define-key map "\M-gat" 'gklee-add-thread-show) ;add filter
	(define-key map "\M-gaw" 'gklee-add-warp-show)
	(define-key map "\M-gaf" 'gklee-add-file-show)
	(define-key map "\M-gab" 'gklee-add-block-show)
	(define-key map "\M-grt" 'gklee-remove-thread) ;remove filter
	(define-key map "\M-grw" 'gklee-remove-warp)
	(define-key map "\M-grf" 'gklee-remove-file)
	(define-key map "\M-grb" 'gklee-remove-block)

	(define-key map "\M-guf" 'gklee-unfilter) ;unfilter
	(define-key map "\M-gta" 'gklee-toggle-asm-visible)
	;Change maybe?
	(define-key map "\M-gss" 'gklee-start-step)
	(define-key map "\M-gne" 'gklee-next-error)
	map))

(defun gklee-next-error ()
"This function will advance the gklee-trace-error-list
-- a circular list -- and set the trace buffer focus
to the next error"
(interactive)
(progn
  (setq gklee-trace-error-list (cdr gklee-trace-error-list))
  (gklee-set-focus-to-error)
  ))

;;filter handling
(defun gklee-toggle-asm-visible()
  (interactive)
  (with-current-buffer gklee-trace-buffer
    (setq invSym (gklee-intern "ASM" gklee-filter-obarray))
    (if gklee-asm-visible
	(progn
	  (setq gklee-asm-visible nil)
	  (gklee-add-to-invisibility-spec invSym)
	  )
      (setq gklee-asm-visible t)
      (remove-from-invisibility-spec invSym)
      )
    ;; (setq redisplay-dont-pause t)
    ;; (redisplay t)
    (redraw-display)
    )
  )


(defun gklee-remove-item(id prefix)
  "This function will add id-prefix to the trace
filter list"
  (with-current-buffer gklee-trace-buffer
    (let ((symbol (gklee-intern (concat prefix (if (stringp id)
					     id
					   (number-to-string id)))
			  gklee-filter-obarray)))
      (gklee-add-to-invisibility-spec symbol))))

(defun gklee-remove-thread(id)
  "This function will add thread id to the trace buffer 
filter list -- thereby hiding it"
  (interactive "nThread: ")
  (gklee-remove-item id "T"))

(defun gklee-remove-block(id)
  "This function will add block id to the trace buffer
filter list -- thereby hiding it"
  (interactive "nBlock: ")
  (gklee-remove-item id "B"))

(defun gklee-remove-file(file)
  "This function will add file id to the trace buffer
filter list"
  (interactive "fFile: ")
  (gklee-remove-item (expand-file-name file) "F"))

;;  (defun gklee-show-locations (locations)
;; "This function will hide all but the locations
;; passed in (in form of '(file:line file1:line1 . . .))"
;; (with-current-buffer gklee-trace-buffer
;;   (gklee-reset-base-invisibility-spec)
;;   (mapatoms
;;    (lambda (atm)
;;      (let ((symName (symbol-name atm)))
;;        (catch 'equ
;;        (mapc (lambda (loc)
;; 	       (if (equal symName loc)
;; 		   (throw 'equ)
;; 		 )
;; 	       )
;; 	     )
;;        (gklee-add-to-invisibility-spec atm))))
;;    gklee-filter-obarray)
;;   ))

(defun gklee-remove-warp(id)
  "This function will add warp id to the trace buffer
filter list"
  (interactive "nWarp: ")
  (let* ((firstT (gklee-get-first-t-in-warp warp))
	 (lastT (gklee-get-last-t-in-warp warp))
	 (iter firstT))
    (while (<= iter lastT)
      (gklee-remove-item iter "T")
      (setq iter (+ iter 1)))))

(defun gklee-reset-base-invisibility-spec ()
  (progn
    (setq buffer-invisibility-spec nil)
    (if (not gklee-asm-visible)
	(gklee-add-to-invisibility-spec (gklee-intern "ASM" gklee-filter-obarray)))))

;; (defun gklee-show-errors(ids)
;;   "This function will clear filter list and 
;; add all but id"
;;   (gklee-show-items ids "C")
;;   )
;;   (with-current-buffer gklee-trace-buffer
;;     (gklee-reset-base-invisibility-spec)
;; ;    (setq buffer-invisibility-spec nil)
;;     (mapatoms 
;;      (lambda (atm)
;;        (let ((symName (symbol-name atm))
;; 	     ;; (convID (if (stringp id)
;; 	     ;; 		 id
;; 	     ;; 	       (number-to-string id)
;; 		       )
;; 	 (if (and 
;; 	      ;; (not (eq 0 atm))
;; 	      ;; (equal (substring  symName 0 (length prefix))
;; 	      ;; 	     prefix)
;; 	      (not (equal symName id)))
;; 	     (gklee-add-to-invisibility-spec atm))))
;; 	 gklee-filter-obarray)
;;        ))

(defun gklee-show-items(ids prefix)
  "This will clear the trace filter and add all items
with same prefix to the filter list except those in ids"
  (with-current-buffer gklee-trace-buffer
    (gklee-reset-base-invisibility-spec)
					;    (setq buffer-invisibility-spec nil)
    (mapatoms 
     (lambda (atm)
       (let ((symName (symbol-name atm))
	     )
	 (catch 'equ
	   (mapc (lambda (loc)
		   (if (and 
			(not (eq 0 atm))
			(equal (substring  symName 0 (length prefix))
			       prefix)
			(equal symName (concat prefix loc)))
		       (throw 'equ nil)
		     )
		   ) ids
		     )
	   (if (equal (substring symName 0 (length prefix))
		      prefix)
	       (gklee-add-to-invisibility-spec atm)))))
     gklee-filter-obarray)
    ))

(defun gklee-show-item(id prefix)
  "This function will clear the trace filter list and filter
out all items of the same type as prefix except id"
  (with-current-buffer gklee-trace-buffer
    (gklee-reset-base-invisibility-spec)
					;    (setq buffer-invisibility-spec nil)
    (mapatoms 
     (lambda (atm)
       (let ((symName (symbol-name atm))
	     (convID (if (stringp id)
			 id
		       (number-to-string id)
		       )))
	 (if (and 
	      (not (eq 0 atm))
	      (equal (substring  symName 0 (length prefix))
		     prefix)
	      (not (equal symName (concat prefix convID))))
	     (gklee-add-to-invisibility-spec atm))))
     gklee-filter-obarray)
    ))

(defun gklee-add-show-item(id prefix)
  "This function will remove prefix+id from the 
trace filter list"
  (with-current-buffer gklee-trace-buffer
    (let* ((convID (if (stringp id)
		       id
		     (number-to-string id)))
	   (symbol (gklee-intern (concat prefix convID) gklee-filter-obarray)))
      (remove-from-invisibility-spec symbol))))

(defun gklee-add-block-show(blk)
  "This interactive will remove a block id from trace filter list"
  (interactive "nBlock: ")
  (gklee-add-show-item blk "B"))

(defun gklee-add-thread-show(thd)
  "This interactive will remove a thread id from trace filter list"
  (interactive "nThread: ")
  (gklee-add-show-item thd "T"))

(defun gklee-add-file-show(file)
  "This function will remove trace lines in file 'file'"
  (interactive "fFile: ")
  (gklee-add-show-item (expand-file-name file) "F"))

(defun gklee-show-thread(thd)
  "This interactive will filter all threads from trace except 
'thd'"
  (interactive "nThread: ")
  (gklee-show-item thd  "T"))

(defun gklee-show-block(blk)
  "This interactive will filter all blocks from trace except
'blk'"
  (interactive "nBlock: ")
  (gklee-show-item blk "B"))

(defun gklee-show-file(file)
  "This function will filter all trace lines of files
different than 'file'"
  (interactive "fFile: ")
  (gklee-show-item (expand-file-name file) "F"))

(defun gklee-show-warp(warp)
  "This interactive will filter all warps from trace except
'warp'"
  (interactive "nWarp: ")
  (let* ((firstT (gklee-get-first-t-in-warp warp))
	 (lastT (gklee-get-last-t-in-warp warp))
	 (iter (+ firstT 1)))
    (gklee-show-item firstT "T")
    (while (<= iter lastT)
      (gklee-add-show-item iter "T")
      (setq iter (+ iter 1)))))

(defun gklee-get-first-t-in-warp(warp)
  (* warp gklee-threads-per-warp)
  )

(defun gklee-get-last-t-in-warp(warp)
  (+ (* warp gklee-threads-per-warp) (- gklee-threads-per-warp 1))
  )

;; (defun gklee-set-filter(blk thd)
;;   (with-current-buffer gklee-trace-buffer
;;     (setq buffer-invisibility-spec (list 
;; 				    (gklee-intern
;; 				     (concat "B" 
;; 					     (number-to-string blk)
;; 					     )
;; 				     )
;; 				    (gklee-intern
;; 				     (concat "T"
;; 					     (number-to-string thd)
;; 					     )
;; 				     )
;; 				    )))
;;   )

(defun gklee-unfilter()
  "This interactive will clear the trace filter list"
  (interactive)
  (with-current-buffer gklee-trace-buffer
    (gklee-reset-base-invisibility-spec)
					;    (setq buffer-invisibility-spec nil)
    ))

;; (defun gklee-filter(blk thd)
;;   (interactive "nBlock:\nnThread:\n")
;;   (gklee-set-filter blk thd)
;; )

(defun gklee-get-current-trace-num()
  "Extracts the number for the current trace from the name of the trace buffer.
This is used to lookup errors when matching them to trace lines.  This 
should not be called if a trace hasn't been selected in *gklee-run*."
  (gklee-get-lineno gklee-trace-buffer)
  )

(defun gklee-trace-has-errors(tname)
  "Decision proceedure that checks gklee-error-htable for errors
for the given trace"
  (catch 'exit 
    (maphash (lambda (key val)
	       (let ((res (string-match "\\(^[0-9]+\\):" key)))
		 (if (and
		      res
		      (equal (string-to-number (match-string 1 key)) 
			     (gklee-get-lineno tname)))
		     (throw 'exit t))))
	     gklee-error-htable)
    ))

(defconst gklee-error-table
  '(
    ("assert" . "Assertion violation")
    ("wwrwb"  . "Write write race within warp benign")
    ("wwrw"   . "Write write race within warp")
    ("wwrawb" . "Write write race across warps benign")
    ("wwraw"  . "Write write race across warps")
    ("rwraw"  . "Read write race across warps")
    ("wwbdb"  . "Write write branch divergence race benign")
    ("wwbd"   . "Write write branch divergence race")
    ("rwbd"   . "Read write branch divergence race")
    ("rw"     . "Write read race")
    ("ww"     . "Write write race")
    ("dlbm"   . "Deadlock due to barrier mismatch\nlocation reported is first thread in divergent set")
    ("dbs"    . "Potential deadlock -- different barrier sequence")
    ("bsdl"   . "Potential deadlock -- barrier sequences of differing length")
    ("wwbc"   . "Write write bank conflict")
    ("rrbc"   . "Read read bank conflict")
    ("rwbc"   . "Read write bank conflict")
    ("mc"     . "Non-coalesced global memory access")
    ("mv"     . "Missing volatile")
    ("wd"     . "Warp divergence")))

(defun gklee-lookup-error-code (code)
  "This function returns a descriptive string of the passed
error code"
  (cdr (assoc code gklee-error-table))
  )

;; (defun gklee-remove-trace (code)
;;   "This removes the trace info from an error code"
;;   (let ((match (string-match "[0-9]*:" code)))
;;     (substring code (match-end 0) (length code)))
;;   )

;; (defun gklee-lookup-filter-error-code (line)
;;   "Takes an error info list: (code loc0 loc1) and
;; returns a ..."
;;   (let (value)
;;     (dolist (item (cdr line) value)
;;       (setq value (cons (gklee-remove-trace item) value))))
;;   )

(defun gklee-deconstruct-location (loc)
  "This function will break up an id string
into a human readable form and return the resulting
string"
  (if loc
      (let* ((split (split-string loc ":")))
	(if (> (length split) 0)
	    (let ((file (gklee-get-file-from-path (nth 2 split)))
		  (line (nth 3 split)))
	      (if (not
		   (and file line))
		  "Location information not available."
		(concat 
		 file
		 " on line: " line
		 ", block: " (nth 0 split)
		 ", thread: " (nth 1 split))
		)))
	))
)

(defun gklee-get-id-string (item err-code)
  "This function produces a string that shows identifying info for 
error to be listed with a trace header in gklee-run"
  (if (equal err-code "dlbm")
      (gklee-deconstruct-location(car (cdr item)))
    (concat (gklee-deconstruct-location(car (cdr item))) 
	    "\n" (gklee-deconstruct-location(car (cdr (cdr item)))))
    ))

(defun gklee-get-error-type-list(err-lists properties)
  (let ((result))
    (dolist (item err-lists result)
      (let ((prop-loc (cons 'gklee-err-info 
				(cons (cdr item) properties))))
	(setq result 
	      (concat
	       (apply 'propertize 
		      (cons (concat "\n" (gklee-lookup-error-code (car item)) "\n")
			    (append prop-loc (list 'face (list :foreground "red")))))
	       (apply 'propertize
		      (cons (gklee-get-id-string item (car item)) prop-loc))
					;gklee-lookup-filter-error-code item) properties)))
			    
		      result))
	      ))))

(defun gklee-list-trace-errors (tname properties)
  "Function returns string of errors associated with the given trace
and propertizes with given properties plus a filter property for each error"
(let ((err-info-lines "")
      (tr-err-list (cadr (assoc (gklee-get-lineno (file-name-nondirectory tname))
				gklee-error-alist)))
      )
  (gklee-get-error-type-list tr-err-list properties)))
  ;; (mapc (lambda (err)
  ;; 	  (let ((descr (assoc (car err) gklee-error-table))
  ;; 		()))))))
;;the old version:
  ;; (catch 'exit 
  ;;   (maphash (lambda (key val)
  ;; 	       (let ((res (string-match "\\(^[0-9]+\\):" key)))
  ;; 		 (if (and
  ;; 		      res
  ;; 		      (equal (string-to-number (match-string 1 key))
  ;; 			     (gklee-get-lineno tname)))
  ;; 		     (throw 'exit 
  ;; 			    (gklee-get-error-type-list val 
						       
  ;; 						       (append 
							
  ;; 							(list 'face 
  ;; 							      (list :foreground "red"))
  ;; 							properties))
  ;; 			    ))))
  ;; 	     gklee-error-htable)
  ;;   )
  ;; )

(defun gklee-reset-trace-source-states(trace-source-list)
  "Restores all source windows associated with a trace
to their previous state.  trace-source-list is an alist in form:
'((buffer-name file was-open was-read-only saved-point)([next rec]))'"
  (let* ((ts-list (car trace-source-list))
	 (vals (cdr ts-list))
	 (buffer (get-buffer (car ts-list)))
	 (was-open (cadr vals))
	 (was-read-only (cadr (cdr vals)))
	 (ts-rest (cdr trace-source-list))
	 (src-point (cadr (cdr (cdr vals)))))
    (if buffer
	(with-current-buffer buffer
	  (if (not was-open)
	      (kill-buffer buffer)
	    (if (not was-read-only)
		(progn
		  (setq buffer-read-only nil)
		  )
	      )
	    (buffer-enable-undo)
	    (set-marker gklee-source-mark src-point)
	    (setq overlay-arrow-position nil)
	    )
	  )
      )
    (if ts-rest
	(gklee-reset-trace-source-states ts-rest)
      )))

(defun gklee-reset-trace-source-state ()
  (if gklee-trace-source-buffers
      (gklee-reset-trace-source-states gklee-trace-source-buffers)))

(defun gklee-exit-trace ()
  "This function causes the run buffer to become active rather
than the trace window"
  (interactive)
  (progn
    (if (get-buffer gklee-active-filter-buffer)
	(kill-buffer gklee-active-filter-buffer))
    (if (get-buffer gklee-available-filter-buffer)
	(kill-buffer gklee-available-filter-buffer))
    (gklee-reset-trace-source-state)
    (set-window-buffer (selected-window) gklee-run-buffer)
    (kill-buffer gklee-trace-buffer)
					;    (kill-buffer gklee-trace-error-buffer)
    (setq gklee-trace-buffer nil)
					;(setq gklee-trace-error-buffer nil)
    ))

(defun gklee-populate-trace-error-list ()
  "This function creates the gklee-trace-error-buffer, or clears the existing 
one, and then loads it with a hierarchical list by
trace, file, line, block and tid"
  (if (get-buffer gklee-trace-error-buffer)
      (kill-buffer gklee-trace-error-buffer))
  (get-buffer-create gklee-trace-error-buffer)
  (gklee-load-trace-error-buff)
  )

(defun gklee-load-trace-error-buff()
  "Loads the trace error buffer so that an error may be 
selected to filter the trace on in order to locate the
problem in the source code"
  (let (;(buf-num (gklee-get-current-trace-num))
	last-file
	last-line  
	sorted-keys
	(indent 0))
    (with-current-buffer gklee-trace-error-buffer
      
      (insert (concat
	       (format "Error list for trace %s by location\n" gklee-trace-buffer)
	       "Click on an error to select\n\n"))
      (maphash (lambda (key val)
		 (setq sorted-keys (cons key sorted-keys))) gklee-error-htable)
      (sort sorted-keys 'string<)
      (mapcar (lambda (key)
		(let* ((value (gethash key gklee-error-htable))
		       (key-comps (split-string key ":"))
		       (file (nth 4 key-comps))
		       (line (nth 3 key-comps))
		       )
		  (if (not (equal last-file file))
		      (progn
			(insert (concat file "\n"))
			(setq last-file file)
			(setq last-line nil)))
		  (if (not (equal last-line line))
		      (progn
			(insert (concat "line: " line "\n"))
			(setq last-line line)))
		  )) sorted-keys
		     ))))

(defun gklee-set-focus-to-error ()
  "This function sets the focus to the current car of 
gklee-trace-error-list"
  (if gklee-trace-error-list
      (with-current-buffer gklee-trace-buffer
	(goto-char (car gklee-trace-error-list))
	(beginning-of-line 2))))
    
 ;;  "This function moves the trace buffer point to the location of
;; the first error, if any"
  
;;   (if gklee-trace-first-err-line
;;       (let ((pos 1))
;; 	(while (setq pos (next-single-property-change pos 'gklee-trace-line))
;; 	  (if (equal gklee-trace-first-err-line 
;; 		     (get-text-property pos 'gklee-trace-line))
;; 	      (progn
;; 		(goto-char pos)
;; 		(setq pos (point-max)) ;this will cause exit
;; 		))))))

(defun gklee-filter-event (e)
"This is called when clicking on a filter item in either
available or active filter buffers"
(interactive "e")
(gklee-filter-item
 (posn-point (event-end e)))
)

(defun gklee-filter-all (sym is-active-buf)
"This function takes one of the 'all' filter
symbols, which represent all of the members of
one of the filter categories (such as blocks)
and then applies all the symbols for that group
to either activate or deactivate its filter
in the trace buffer (and do the bookkeeping
required for showing / hiding in active/available
filter buffers"
  (if (equal (symbol-name sym) "All_Blocks")
      (setq sym-list gklee-block-syms)
    (if (equal (symbol-name sym) "All_Threads")
	(setq sym-list gklee-thread-syms)
      (if (equal (symbol-name sym) "All_Files")
	  (setq sym-list gklee-file-syms)
	(if (equal (symbol-name sym) "All_Locations")
	    (setq sym-list gklee-loc-syms)
	  (if (equal (symbol-name sym) "All_Warps")
	      (progn
		(gklee-filter-all (gklee-intern "All_Threads" gklee-filter-obarray) is-active-buf)
		(setq sym-list gklee-warp-syms)
		)
	      )))))
  (if is-active-buf
      (progn
	(with-current-buffer gklee-active-filter-buffer
	  (mapc 'add-to-invisibility-spec sym-list))
	(with-current-buffer gklee-available-filter-buffer
	  (mapc 'remove-from-invisibility-spec sym-list))
	(with-current-buffer gklee-trace-buffer
	  (mapc 'remove-from-invisibility-spec sym-list)))
    (with-current-buffer gklee-active-filter-buffer
      (mapc 'remove-from-invisibility-spec sym-list))
    (with-current-buffer gklee-available-filter-buffer
      (mapc 'add-to-invisibility-spec sym-list))
    (with-current-buffer gklee-trace-buffer
      (mapc 'add-to-invisibility-spec sym-list)))
  )

(defun gklee-get-invis-tag-id (sym &optional prefix-len)
"Takes a symbol used for invisibility list
filtering, and returns the numerical id for it"
(let ((pre-len (if prefix-len prefix-len 1)))
  (string-to-number (substring (symbol-name sym) pre-len (length (symbol-name sym))))
  )
)

(defun gklee-get-threads-by-warp (warp threads)
(let* ((warp-id (gklee-get-invis-tag-id warp))
       (first-thread (gklee-get-first-t-in-warp warp-id))
       (last-thread (gklee-get-last-t-in-warp warp-id))
       (out-list))
  (mapcar (lambda (thd)
	    (let ((cw (gklee-get-invis-tag-id thd)))
	      (if (and
		   (not (equal "All_Threads" (symbol-name thd)))
		   (<= cw last-thread)
		   (>= cw first-thread))
		  thd
		)))
	  threads))
)

(defun gklee-remove-from-invisibility-spec (item)
  (if item
      (remove-from-invisibility-spec item))
)

(defun gklee-add-to-invisibility-spec (item)
  (if item
      (add-to-invisibility-spec item))
  )

(defun gklee-filter-warp (sym in-active-buf)
  "This takes a warp symbol and then filters/unfilters all threads
related to that symbol, depending upon whether this is being
called under the 'active' filter buffer or the 'available' filter
buffer.  This function handles all bookkeeping wrt the visibility
of the symbol in the filter buffers"
  (let ((threads (gklee-get-threads-by-warp sym gklee-thread-syms)))
    (if in-active-buf
	(progn
	  (with-current-buffer gklee-active-filter-buffer 
	    (mapc 'gklee-add-to-invisibility-spec threads))
	  (with-current-buffer gklee-available-filter-buffer
	    (mapc 'gklee-remove-from-invisibility-spec threads))
	  (with-current-buffer gklee-trace-buffer
	    (mapc 'gklee-remove-from-invisibility-spec threads))
	  )
      (with-current-buffer gklee-active-filter-buffer
	(mapc 'gklee-remove-from-invisibility-spec threads))
      (with-current-buffer gklee-available-filter-buffer
	(mapc 'gklee-add-to-invisibility-spec threads))
      (with-current-buffer gklee-trace-buffer
	(mapc 'gklee-add-to-invisibility-spec threads))))
  )
 
 
(defun gklee-filter-item (pos)
  "This function takes a point from a filter buffer
and performs the filter operation (shuffling filter
symbols around buffer-invisibility-spec's)"
(let* ((buf (buffer-name))
      (isym (get-text-property pos 'invisible))
      (in-active (equal gklee-active-filter-buffer buf))
      (sym-name (symbol-name isym))
      )
  (gklee-add-to-invisibility-spec isym)
  (if (and
       (> (length sym-name) 2)
       (equal (substring (symbol-name isym) 0 3) "All"))
      (gklee-filter-all isym in-active)
    (if (equal (substring (symbol-name isym) 0 1) "W")
	(gklee-filter-warp isym in-active)))
  (if in-active
      (progn
	(with-current-buffer gklee-trace-buffer
	  (remove-from-invisibility-spec isym))
	(with-current-buffer gklee-available-filter-buffer
	  (remove-from-invisibility-spec isym))
	)
    (with-current-buffer gklee-trace-buffer
      (gklee-add-to-invisibility-spec isym))
    (with-current-buffer gklee-active-filter-buffer
      (remove-from-invisibility-spec isym)
      ))
  (redraw-display))) ;;redraw-display is a hack to get the buffers to update after filtering

(defun gklee-insert-symbols-with-invis (symlist sep invis &optional is-path)
  "Performs an insert (with current buffer) of symlist, 
separated by 'sep', adding itself
as an 'invisible symbol -- the symbols MUST be interned in 
gklee-filter-obarray already"
  (let ((i-list (if is-path
		    (list 'invisible (gklee-intern "HIDDEN" gklee-filter-obarray))
		  nil)))
	(mapc (lambda (sym)
		(let ((sym-name (symbol-name sym))
		      (reg-i-list (list 'invisible sym 'category 'gklee-filter-cat)))
		  (insert
		   (concat
		    (if (and is-path (string-match "/" sym-name))
			(progn
			  (apply 'propertize 
				 (cons
				  (file-name-directory sym-name)
				  i-list))
			  (apply 'propertize
				 (cons
				  (file-name-nondirectory sym-name)
				  reg-i-list))
			  )
		      (apply 'propertize
			     (cons sym-name reg-i-list))
		      )
		    (propertize sep 'invisible sym)
		    )
		   ))
		(if invis
		    (gklee-add-to-invisibility-spec sym)));;makes it invisible in current buffer
	      symlist)
	)
    )

(defun gklee-fill-filter-buffer (buf blk thd fil loc wrp)
  "This will load the passed buffer with tags for each
category of item to filter"
  (let ((make-invis (equal buf gklee-active-filter-buffer)))
    (with-current-buffer buf
      (erase-buffer)
      (use-local-map gklee-filter-keymap)
      (setq buffer-invisibility-spec (list (gklee-intern "HIDDEN"
							 gklee-filter-obarray)))
      (insert (propertize "Block Filters\n" 'face 'bold))
      (gklee-insert-symbols-with-invis blk "  " make-invis)
      (insert (propertize "\n\nThread Filters\n" 'face 'bold))
      (gklee-insert-symbols-with-invis thd "  " make-invis)
      (insert (propertize "\n\nFile Filters\n" 'face 'bold))
      (gklee-insert-symbols-with-invis fil "  " make-invis t)
      (insert (propertize "\n\nWarp Filters\n" 'face 'bold))
      (gklee-insert-symbols-with-invis wrp "  " make-invis)
      (insert (propertize "\n\nLocation Filters\n" 'face 'bold))
      (gklee-insert-symbols-with-invis loc "  " make-invis t)
      (goto-char 0)
      )
    )
  )

(defun gklee-load-warps (w)
"This function loads the warps list based the number
of threads that were encountered by GKLEE and 
the setting of 'gklee-threads-per-warp"
(let ((count 0))
  (while (< count gklee-warp-count)
    (add-to-list w (gklee-intern (concat "W" (number-to-string count))
				gklee-filter-obarray))
    (setq count (1+ count))
    ))
)

(defun gklee-clear-filter-syms()
  (setq gklee-block-syms nil)
  (setq gklee-thread-syms nil)
  (setq gklee-file-syms nil)
  (setq gklee-loc-syms nil)
  (setq gklee-warp-syms nil)
)

(defun gklee-setup-filter-buffers ()
  "This function will populate the filter buffers
-- gklee-active-filter-buffer and gklee-available-filter-buffer --
with the filter items.  They always exist in both buffers,
but the filter item identifier is a member of its own
invisibility list.  When a filter is switched from
active to available [and similarly for vice-versa], 
its identifier [a symbol, also stored in gklee-filter-obarray,
the object vector in which we maintain the filter symbols]
is removed from the available buffer's
buffer-invisibility-spec and added to the active buffer's
b-i-s -- also, the identifier is removed from the trace
buffer's buffer-invisibility-spec"
					;to begin, we make a list of each type to sort [so that they can be placed
					;in an ordered fashion in the buffers]
  (get-buffer-create gklee-active-filter-buffer)
  (get-buffer-create gklee-available-filter-buffer)
  (gklee-clear-filter-syms)
  (mapatoms 
   (lambda (sym)
     (if (not (equal 0 sym)) ;this is important, as empty items in sym vect are 0
	 (if (equal "B" (substring (symbol-name sym) 0 1))
	     (setq gklee-block-syms (cons sym gklee-block-syms))
	   (if (equal "T" (substring (symbol-name sym) 0 1))
	       (setq gklee-thread-syms (cons sym gklee-thread-syms))
	     (if (equal "F" (substring (symbol-name sym) 0 1))
		 (progn
		   ;(debug)
		   (setq gklee-file-syms (cons sym gklee-file-syms))
		   )
	       (if (equal "C" (substring (symbol-name sym) 0 1))
		   (setq gklee-loc-syms (cons sym gklee-loc-syms))
		 ))))
       )) gklee-filter-obarray)
  (setq gklee-block-syms (sort gklee-block-syms 'string<))
  (setq gklee-thread-syms (sort gklee-thread-syms 'string<))
  (setq gklee-file-syms (sort gklee-file-syms 'string<))
  (setq gklee-loc-syms (sort gklee-loc-syms 'string<))
  (gklee-load-warps 'gklee-warp-syms)
  (gklee-add-all-filter 'gklee-block-syms "Blocks")
  (gklee-add-all-filter 'gklee-thread-syms "Threads")
  (gklee-add-all-filter 'gklee-file-syms "Files")
  (gklee-add-all-filter 'gklee-loc-syms "Locations")
  (gklee-add-all-filter 'gklee-warp-syms "Warps")
					;now load buffers
  (gklee-fill-filter-buffer gklee-active-filter-buffer gklee-block-syms gklee-thread-syms
			    gklee-file-syms gklee-loc-syms gklee-warp-syms)
  (gklee-fill-filter-buffer gklee-available-filter-buffer gklee-block-syms gklee-thread-syms
			    gklee-file-syms gklee-loc-syms gklee-warp-syms)
  )


(defun gklee-add-all-filter (list name)
  "This will add an 'All' item for selecting everything in
a category at once"
  (progn
    ;; (message (concat "entered -add-all-filter with " name))
    ;; (mapc (lambda (x) (message (symbol-name x))) (symbol-value list))
    (add-to-list list (gklee-intern (concat "All_" name) gklee-filter-obarray))
    ;; (message "Exiting -add-all-filter with:")
    ;; (mapc (lambda (x) (message (symbol-name x))) (symbol-value list))
    )
  )

(defun gklee-follow-trace (p)
  "This function will open a trace that is selected
in *gklee-run*, populate a buffer with the trace lines,
and close the previous trace buffer (if any)"
  (let ((b (get-text-property p 'gklee-trace-buffer))
	(tr (get-text-property p 'gklee-trace-file))
	(err-id (get-text-property p 'gklee-err-info))
	)
;    (gklee-exit-trace)
    (if gklee-trace-buffer
    	(kill-buffer gklee-trace-buffer))
    (setq gklee-trace-error-list nil)
    (setq gklee-trace-buffer b)
    (gklee-process-trace b tr err-id)
    (set-window-buffer (selected-window) b)
    (with-current-buffer gklee-trace-buffer
      (setq buffer-invisibility-spec nil)
      (gklee-set-focus-to-error)
      )
    (gklee-setup-filter-buffers)
    (setq gklee-asm-visible t)
    (gklee-create-windows gklee-source-buffer gklee-trace-buffer gklee-available-filter-buffer gklee-active-filter-buffer)
     (with-current-buffer gklee-source-buffer
       (local-set-key [mouse-3] 'gklee-source-right-click))
     ;(gklee-
;    (gklee-set-current-trace-line) ;;FOR WORK ON STEP MODE
    ))

(defun gklee-set-current-trace-line ()
  "Sets the current trace line for stepping
-- this needs work"
  (let ((end 0)
	(begin 0))
    (with-current-buffer gklee-trace-buffer
      (beginning-of-buffer)
      ;;    (setq overlay-arrow-position (point-marker)) ;;this won't work -- seems to be only one
      (gklee-goto-line (point))
      (setq begin (point))
      (setq end (next-single-property-change begin 'gklee-trace-line))
      (setq end (1- end))
      (set-text-properties begin end '(face highlight))
      )
    )
)

(defun gklee-follow-trace-key ()
  "This function calls gklee-follow-trace with info provided by keyboard
navigation of a path in gklee-run window"
  (interactive)
  (gklee-follow-trace (point)))

(defun gklee-follow-trace-event (e)
  "This function calls gklee-follow-trace with info provided by
mouse navigation of a path (trace) in gklee-run window"
  (interactive "e")
  (gklee-follow-trace (posn-point (event-end e))))

(put 'gklee-trace-link-cat 'keymap
     (let ((map (make-sparse-keymap)))
       (define-key map [mouse-1] 'gklee-follow-trace-event)
       (define-key map "\^M" 'gklee-follow-trace-key)
       map))

(put 'gklee-trace-link-cat 'mouse-face 'highlight)


(defun gklee-get-name-from-path (path)
  (let* ((files (split-string path "/" t))
	 (len (length files)))
    (if (> len 0)
	(nth (- len 1) files)
      nil)))

(defun gklee-record-source-state (source-buffer file was-open point)
  "This function will add the current state of the source
buffer to the trace-source-buffers  (if not already
present"
  (with-current-buffer source-buffer
    (let ((values (assoc (buffer-name) gklee-trace-source-buffers)))
      (if (not values)
	  (setq gklee-trace-source-buffers 
		(append gklee-trace-source-buffers 
			(list 
			 (cons (buffer-name) 
			       (list file was-open buffer-read-only point)))))
	;; (cons (list (buffer-name) file was-open buffer-read-only)
	;;       gklee-trace-source-buffers)
	))))

(defun gklee-goto-line (pos); source-buffer)
  "This function reads the 'gklee-line property of the
current trace window at point pos, selects the source buffer,
and then sets focus on the line pointed to and
the 'overlay-arrow-position', the margin arrow"

  (let* ((line (get-text-property pos 'gklee-line
				  (current-buffer)))
	 (file (get-text-property pos 'gklee-source
				  (current-buffer))))
    (if(and file line)
	(let* (
	       (source-buffer-name (gklee-get-name-from-path file))
	       (was-open (get-buffer source-buffer-name))
	       (source-buffer (find-file-noselect file));(get-buffer-create source-buffer-name))
	       (window (get-buffer-window source-buffer))
	       (source-mark gklee-source-mark))
	  (if line
	      (with-current-buffer source-buffer
		(display-buffer source-buffer)
		(save-selected-window
		  (select-window (get-buffer-window source-buffer))
		  (goto-line line)
		  (set-marker gklee-source-mark (point))
		  (setq overlay-arrow-position gklee-source-mark)
		  (gklee-record-source-state source-buffer file was-open (point))
		  (setq buffer-read-only nil)
		  ))
	    (message "There is no line information for this event"))))))

(defun gklee-goto-line-key ()
  "This is called when the key selected to activate a line
in a trace buffer is hit -- calls gklee-goto-line"
  (interactive)
  (gklee-goto-line (point))); gklee-source-buffer))

(defun gklee-get-opt-level (n)
  "Prompts user for optimization level to compile at"
  (interactive "nOptimization level [0,1,2,3]: ")
  (if (and (< n 4)(> n -1))
      (setq gklee-opt-level (concat "-O" (number-to-string n)))
    (message "You entered %d, using default level of 0" n)
    (setq gklee-opt-level "-O0")
    ))

(defun gklee-goto-line-event (event)
  "This is called when the user clicks on a trace line
with the mouse -- calls gklee-goto-line"
  (interactive "e")
  (gklee-goto-line
   (posn-point (event-end event)))); gklee-source-buffer))

(defun gklee-partial-refresh ()
  "This function adds statistics to the gklee-run window
before Gklee has completed execution"
  (erase-buffer)
  (insert 
   (format
    "Running GKLEE . . .\n\t Paths Explored: %d"
    gklee-process-record-count)))

(defun gklee-compile ()
  "This performs the llvm execution of the selected
c/c++ source buffer"
  (interactive)
  (let ((buffer (get-buffer-create gklee-compile-buffer-name))
	(process-connection-type nil)
	(file-name (buffer-file-name))
	(opt-level (call-interactively 'gklee-get-opt-level)))
    (gklee-create-windows (buffer-name) gklee-run-buffer-name gklee-run-debug-buffer-name gklee-compile-buffer-name)
    (if file-name
	(let ((args (append (list
			     (concat gklee-home-path gklee-compile-bin)
			     "-o"
			     (concat (expand-file-name
				      (or gklee-comp-out-path
					  gklee-default-path))
				     "target.o")
			     file-name
			     opt-level)
			    (append gklee-compile-args
				    gklee-user-compile-args))))
	  (with-current-buffer buffer
	    (erase-buffer)
	    (buffer-disable-undo)
	    (insert "executed with:\n")
	    (insert (mapconcat 'identity args " "))
	    (insert "\n\nin directory:\n"
		    (shell-command-to-string "pwd"))
	    (insert "\n\ncompilation output:\n"))
	  (let ((process (apply 'start-process
				gklee-compile-buffer-name
				gklee-compile-buffer-name
				args)))
	    (setq gklee-compile-process process)
	    process))
      (message "The file doesn't exist")
      nil)))

(defun gklee-get-file-from-path (path)
  (let ((split (split-string path "/")))
    (if (> (length split) 1)
	(nth (- (length split) 1) split)
      nil
      )
    ))

(defun gklee-buffer-string (trace)
  "This function retrieves the contents of a
Gklee .trace file"
  ;;  (with-current-buffer tmp-buffer
  (with-temp-buffer
    (erase-buffer)
    (insert-file-contents trace)
    (buffer-string)
    ))

(defun gklee-is-err-line (blk thd ln file err-id)
"Checks if blk:thd:file:ln is err-id"
(catch 'exit
;; (if (equal ln "80")
;;     (debug))
 (dolist (line err-id)
    (let ((split (split-string line ":")))
      (if (and
	   (equal (nth 0 split) blk)
	   (equal (nth 1 split) thd)
	   (equal (nth 2 split) file)
	   (equal (nth 3 split) ln)
	   )
	  ;; (progn
	  ;;   (message "they're equal")
	  ;;   t
	  ;;   )
	  (throw 'exit t)
	)))
 )
)

(defun gklee-intern (name sym-vect)
"This is a binding on 'intern' for debug purposes"
  ;; (if (equal name "B1")
  ;;     (progn
  ;; 	(debug)
  ;; 	(message (concat "Hit on " name " for gklee-intern debug"))
  ;; 	)
    (intern name sym-vect)
    ;; )
)

(defun gklee-process-trace (b trace err-id)
  "This reads the contents of a .trace file and creates
a buffer for the trace, writing the information to 
display for each line and embedding properties
for the display of line along with the source line
it points to"
  (let* ((entries (split-string (gklee-buffer-string trace) "\\*\\*\\*\\*\\*" t))
	 (vsize (length entries))
					;(source gklee-source-buffer)
	 (trace-buffer (get-buffer-create b))
	 (tr-line-count 0)
	 )
    (with-current-buffer b
      ;;(setq gklee-source-buffer source)
      ;;(setq gklee-run-buffer buffer)
      
      (setq buffer-read-only nil)
      (erase-buffer)
      (buffer-disable-undo)
      (use-local-map gklee-trace-keymap)
      (setq prevBlk "")
      (setq prevThd "")
      (setq prevLine "")
      (setq prevFile "")
      ;;(setq gklee-trace-first-err-line nil)
      ;(setq tr-line-count 0)
					;(setq gklee-trace-seq-array (make-vector (length vsize) 'e))
      (dotimes (i vsize)
	(if (string-match "\\([0-9]+\\):\\([0-9]+\\)\n\\([0-9]+\\):\\([^\\:]*\\):\\([[:graph:][:space:]]*\\)" (car entries))
	    (progn
	      (setq blk (match-string 1 (car entries)))
	      ;;remove me
	      (setq thd (match-string 2 (car entries)))
	      (setq line (match-string 3 (car entries)))
	      (setq file (match-string 4 (car entries)))
	      (setq llvm (match-string 5 (car entries)))
	      (setq invis-list (append 
				(list 
				 (gklee-intern 
				  (concat "B" blk)
				  gklee-filter-obarray
				  )
				 (gklee-intern
				  (concat "T" thd)
				  gklee-filter-obarray
				  )
				 (gklee-intern
				  (concat "F" file)
				  gklee-filter-obarray
				  )
				 (gklee-intern
				  (concat "C" file ":" line)
				  gklee-filter-obarray
				  )
	 			 )
				;; (gklee-get-trace-error-symbols b blk thd line file)
				))
	      ;; (when (not 
	      ;; 	     (and
	      ;; 	      ;(equal blk prevBlk)
	      ;; 	      ;(equal thd prevThd)
	      ;; 	      (equal line prevLine)
	      ;; 	      (equal file prevFile)))
	      (setq is-error (gklee-is-err-line blk thd line file err-id))
	      (if is-error 
		  (setq gklee-trace-error-list (cons (point) gklee-trace-error-list)))
		  ;; (progn
		  ;;   (if (not gklee-trace-first-err-line)
		  ;; 	(setq gklee-trace-first-err-line tr-line-count))
		  ;;   (setq gklee-trace-error-lines tr-line-count)
		  ;;   ))
	      (setq err-props (list 'face (list :foreground "red")))
	      (if (not
		   (and
		    (equal line prevLine)
		    (equal file prevFile)))
		  (progn
		    (setq props (list 			  
				 'category 'gklee-line-cat
				 'gklee-line (string-to-number line)
				 'gklee-source file
				 ;'gklee-trace-line tr-line-count
				 ))
		    (setq locVisitedList (cons (concat blk ":" thd) nil))
		    (insert (concat
			     (propertize
			      "\n"
			      'invisible invis-list
			      )
			     (apply 'propertize
				    (append (cons 
					     (format "Line %s, Block %s, Thread %s, File %s\n" 
						     line blk thd (file-name-nondirectory file))
					     (if is-error
						 (append props err-props)
					       props))
					    (list 'invisible invis-list))))))
		(if (not 
		     (and
		      (equal thd prevThd)
		      (equal blk prevBlk)))
		    (progn
		      (setq local-invis 
			    (if (not (member (concat blk ":" thd) locVisitedList))
				(progn
				    (setq locVisitedList (cons (concat blk ":" thd) locVisitedList))
				    invis-list)
			      (append invis-list (list (gklee-intern "ASM"
								     gklee-filter-obarray
								     )))))
		      (insert (concat
			       (propertize
				"\n"
				'invisible local-invis)
			       (apply 'propertize
				      (append (cons 
					       (format "Line %s, Block %s, Thread %s, File %s\n" 
						       line blk thd (file-name-nondirectory file))
					     (if is-error
						 (append props err-props)
					       props))
					      (list 'invisible local-invis)))))
		      )))
	      (insert
	       (concat
		(apply 'propertize
		       (append (cons llvm 
				     (if is-error
					 (append props err-props)
				       props))
			       (list 'invisible (append invis-list (list (gklee-intern "ASM"
										       gklee-filter-obarray))))))))
	      (setq prevBlk blk)
	      (setq prevThd thd)
	      (setq prevLine line)
	      (setq prevFile file)
;	      (setq tr-line-count (1+ tr-line-count))
	      )
	  )
	(setq entries (cdr entries))
	)
      (setq buffer-read-only t)
      
      (beginning-of-buffer)
      ))
  (gklee-get-warp-count)
  (setq gklee-trace-error-list (reverse gklee-trace-error-list))
  (gklee-circularize-list gklee-trace-error-list)
  )

(defun gklee-get-warp-count()
  "This function is called whenever a new trace is read in.  It
calculates the number of warps per block by the max read thread id"
  (let ((maxThread 0))
    (mapatoms
     (lambda (sym)
       (let* (
	      (id (symbol-name sym))
	      (num 0))
	 (if (string-match "T\\([0-9]+\\)" id)
	     (let ((num (string-to-number (match-string 1 id))))
	       (if (>  num maxThread)
		   (setq maxThread num))))))
     gklee-filter-obarray)
    (setq gklee-warp-count (+ (/ maxThread gklee-threads-per-warp) 1))))

(defun gklee-get-lineno (trace)
  "This takes some text and returns the
first number found as a number"
  (let ((tr trace))
    (if (string-match "klee-last.*\\([0-9]+\\)" tr)
	(string-to-number (match-string 1 tr))
      (progn
	(if (string-match "[0-9]+" tr)
	    (string-to-number (match-string 0 tr))))
)))

(defun gklee-lookup-stat (num list)
  "Looks up the nth sublist from a list of lists,barray)
i.e. ((0 subList)(1 sublist)...(n sublist))"
  (let ((n num)
	(slist list))
    (while (and slist (/= (caar slist) n))
      (setq slist (cdr slist)))
    (cdar slist)))


;;this function will return a string with a line for each
;;statistic collected for the path represented
;;for BC, MC and WD . . .
(defun gklee-get-stat-list (trace)
  "Gets the statistics for a selected (path) trace
to be displayed for it (usually in gklee-run window)"
  (let* ((lineno (gklee-get-lineno trace))
	 (rc (gklee-lookup-stat lineno gklee-race-info-list))
	 (as (gklee-lookup-stat lineno gklee-assertion-info-list))
	 (dl (gklee-lookup-stat lineno gklee-deadlock-info-list))
	 (bc (gklee-lookup-stat lineno gklee-bankcon-info-list))
	 (mc (gklee-lookup-stat lineno gklee-memcol-info-list))
	 (wd (gklee-lookup-stat lineno gklee-warpdiv-info-list))
	 )
    (concat "\n" rc as dl bc mc wd "\n")
    ))

(defun gklee-get-buff-name (trace)
  "Gets the name of the buffer that will contain
the passed trace"
  (concat "*" (gklee-get-name-from-path trace) "*"))

(defun gklee-insert-trace (trace)
  "Given a trace, this will write the trace
data (name and gklee stats) as an entry in the current 
window (normally gklee-run) and
embed properties
to point to source buffer and line"
  (let* ((tr trace)
	 (stats (gklee-get-stat-list trace))
	 (tbuff-name (gklee-get-buff-name trace))
	 (prop-list (list 'category 'gklee-trace-link-cat 'gklee-trace-buffer tbuff-name
			  'gklee-trace-file tr))
	 )
    (progn
      (insert
       (concat
	(apply 'propertize (cons tbuff-name (append (list 'face 'bold)
						    prop-list)))
	;;(apply 'propertize (cons "\n" prop-list))
	(gklee-list-trace-errors tbuff-name prop-list)
	(apply 'propertize (cons (concat "\n" stats) prop-list))
	"\n")))))

(defun gklee-remove-trailing-whitespace (str)
  "Takes a string and returns a string with no trailing
whitespace"
  (when (string-match "[ \t\n]*$" str)
    (substring str 0 (match-beginning 0))))

(defun gklee-insert-buffer-list()
  "This reads all the trace files in the 'klee-last' directory
below the source code file, and makes an entry for each
one in the gklee-run buffer"
  (let* ((cwd (concat gklee-source-path "klee-last/"));(concat (gklee-remove-trailing-whitespace (shell-command-to-string "pwd")) "/klee-last/"))
	 (traces (directory-files cwd))
	 traceList
	 )
    (mapc 
     (lambda (el)
       (if (string-match ".trace" el)
	   (push el traceList)
	 ))
     traces)
    (if traceList
	(progn
	  (newline)
	  (mapc (lambda (trace)
		  (gklee-insert-trace (concat cwd trace))) ;(concat "*gklee-" trace "*"))
		traceList)
	  ))))

(defun gklee-full-refresh()
  "This refreshes the gklee-run window after Gklee has completed
its execution"
  (erase-buffer)
  (insert gklee-info-notice)
  (newline)
  (insert (mapconcat 'identity gklee-summary-info-list "\n"))
  (newline)
  (gklee-insert-buffer-list)
  (goto-char 0))

(defun gklee-kill () 
  "This kills any active Gklee processes [i.e. compile and run]"
  (interactive)
  (if (y-or-n-p "Kill Gklee? ")
      (progn
	(if gklee-compile-process (delete-process gklee-compile-process))
	(if gklee-run-process 
	    (progn
	      (delete-process gklee-run-process)
	      (with-current-buffer gklee-run-buffer
		(push "**GKLEE WAS USER TERMINATED**" gklee-summary-info-list)
		(setq buffer-read-only nil)
		(gklee-full-refresh)
		(setq buffer-read-only t)
		(gklee-reset-trace-source-state)
		)))
	)))

(defun gklee-sentinel (process output)
  "This is the execution sentinel for the Gklee process"
  (if (equal 'exit (process-status process))
      (if (not (equal 0 (process-exit-status process)))
	  (message "gklee: test failed to run\n")
	(let ((buffer (process-buffer process))
	      )
	  (if (not (buffer-live-p buffer))
	      nil
	    (progn
	      (with-current-buffer buffer
		(let ((buffer-read-only nil))
		  (gklee-full-refresh)
		  ))
	      ;(gklee-populate-trace-error-list)
	      )
	    )))))

(defun gklee-run-test (process event) 
  "This function is the sentinel of gklee-compile-process from (interactive) gklee-run,
and will execute Gklee upon the successful termination of gklee-compile-process."
  (if (equal 'exit (process-status process))
      (if (not (equal 0 (process-exit-status process)))
	  (message (concat "gklee: " gklee-compile-bin " compilation failed\n"))
	(progn
	  (with-current-buffer gklee-compile-buffer-name
	    (insert "\nDone.\n"))
	  (let ((buffer (get-buffer-create gklee-run-buffer-name))
		(buffer-debug (get-buffer-create gklee-run-debug-buffer-name))
		(source-buffer (current-buffer))
		)
	    (setq gklee-source-path (file-name-directory (buffer-file-name source-buffer)))
	    (with-current-buffer buffer
	      (use-local-map
	       (let ((map (make-sparse-keymap)))
		 (define-key map "k" 'gklee-kill)
		 map))
	      (setq gklee-source-mark (point-marker)) ;;sets the buffer property of gklee-source-mark?  may not be necessary
	      (setq gklee-source-buffer source-buffer)
	      (gklee-reset-run-state)
	      (erase-buffer)
	      (buffer-disable-undo)
	      (setq buffer-read-only t))
	    (let ((args
		   (append
		    (list (concat gklee-home-path "bin/gklee"))
		    gklee-run-user-args
		    (list 
		     "--trace"
		     "--emacs"
		     )
		    (mapcar (lambda(x)
			      (let ((key (car x))
				    (val (cdr x)))
				(concat key "=" 
					(if(stringp val)
					    val
					  (number-to-string val)))))
			    gklee-options-alist)
		    (append
		     (list (concat (expand-file-name
				    gklee-default-path)
				   "target.o")
			   )
		     gklee-user-program-args)
		    )))
	      (with-current-buffer gklee-run-debug-buffer-name
		(erase-buffer)
		(buffer-disable-undo)
		(insert "executed with: \n")
		(insert (mapconcat 'identity args " "))
		(insert "\n\ngklee output:\n"))
	      (let* ((process-connection-type nil)
		     (process (apply 'start-process
				     gklee-run-buffer-name
				     gklee-run-buffer-name
				     args
				     )))
		(setq gklee-run-process process)
		(set-process-sentinel process 'gklee-sentinel)
		(set-process-filter process 'gklee-run-filter) 
		(switch-to-buffer-other-window buffer)
		)))))))

(defun gklee-set-home-path ()
  "This function attempts to lookup the location of the Gklee installation,
and set 'gklee-home-path' returns nil of not found, the path otherwise"
  (if (not gklee-home-path)
      (let (
	    (gklee-hp (getenv "FLA_KLEE_HOME_DIR"))
	    )
	(if (not gklee-hp)
	    nil
	  (setq gklee-home-path (concat gklee-home-path gklee-hp "/"))
	  ))
    gklee-home-path
    ))


(defun gklee-run ()
  "This is the main interactive function for gklee-mode.  Evaluating this 
function will cause Gklee to compile the active buffer into llvm bytecode 
and then analyze it by passing it to a Gklee invocation."
  (interactive)
  (if (not (gklee-set-home-path))
      (message (concat "You must set environment FLA-KLEE-HOME-DIR "
		       "to location of gklee"))
    (let ((compile-process (gklee-compile)))
      (if compile-process
	  (set-process-sentinel compile-process 'gklee-run-test))))
  )

(defun gklee-insert-debug ()
  "This function drops the contents of 'gklee-record' into the gklee debug buff"
  (let ((record gklee-record))
    (with-current-buffer (get-buffer gklee-run-debug-buffer-name)
      (insert 
       (concat record "\n")
       ))))

(defun gklee-extract-symbols (err-list)
  "This takes a list of lists '((errcode loc-key0 loc-key1 ...)(...))
and returns a list of symbols to identify errors for insertion to
trace line invisibility list"
  (mapcar (lambda (x) (gklee-intern (concat (car x) ":" (car (cdr x)))
			      gklee-filter-obarray)) test-err-list)
  )

;; this is obsoleted -- restructured the error handing system, will
;; iterate through the errors and lookup the trace lines to set highlight
;; properties
;; (defun gklee-get-trace-error-symbols (trace blk thd line file)
;;   "This function checks gklee-error-htable to see if the current trace line
;; is involved in an error condition, and if so, returns a list of symbols for insertion to the
;; invisibility list for identification"
;;   (if (setq err-list (gethash (concat (number-to-string (gklee-get-lineno trace)) ":" 
;; 				      blk ":" thd ":" line ":" file) gklee-error-htable))
;;       (gklee-extract-symbols err-list))
;;   )

(defun gklee-get-error-list (elist ecode)
  (let ((list (car elist))
	(rest (cdr elist)))
    (while (not (equal (car list) ecode))
      (setq list (car rest))
      (setq rest (cdr rest)))
    list)
  )

(defun gklee-string-in-list (str list)
  (let ((match nil))
    (mapc (lambda (s)
	    (if (equal str s)
		(setq match s)))
	  list)
    match)
  )

(defun gklee-add-loc-to-error-list (loc elist)
  (if (not (gklee-string-in-list loc elist))
      (nconc elist (list loc)))
  )

(defun gklee-add-err-list (list code loc0 loc1)
  (nconc list (list (list code loc0 loc1)))
  ) 

(defun gklee-add-trace-error-info (elist)
  "This function adds item to the gklee-error-alist for decorating
*gklee-run* and the relevant trace lines with error information.
elist is:
0     1    2      3       4     5     6      7       8     9
emacs:code:block0:thread0:file0:line0:[block1:thread1:file1:line1]
 -- location 1 is optional"
  (let* (
	(err-code (nth 1 elist))
	(trace-no (if (equal err-code "dlbm") gklee-process-record-count (1+ gklee-process-record-count)))
	(loc0 (concat (nth 2 elist) ":" (nth 3 elist) ":" (nth 4 elist)
		      ":" (nth 5 elist)))
	(loc1 (if (nth 6 elist)
		  (concat (nth 6 elist) ":" (nth 7 elist) ":" (nth 8 elist)
			  ":" (nth 9 elist))
		""
		))
	(trace-err-list (assoc trace-no gklee-error-alist))
	)
    (if trace-err-list
	(nconc (cadr trace-err-list) (list (list err-code loc0 loc1)))
      (if gklee-error-alist
	  (nconc gklee-error-alist (list (list trace-no (list (list err-code loc0 loc1)))))
	(setq gklee-error-alist (list (list trace-no (list (list err-code loc0 loc1)))))
      ))))
    
;; The old -add-trace-error-info
;;  (let* ((loc0 (concat (number-to-string (1+ gklee-process-record-count)) ":" (nth 4 elist) ":" 
;; 		       (nth 5 elist) ":" (nth 2 elist) ":" (nth 3 elist)))
  ;; 	 (entry0 (gethash loc0 gklee-error-htable))
  ;; 	 (loc1 (concat (number-to-string (1+ gklee-process-record-count)) ":" (nth 8 elist) ":" 
  ;; 		       (nth 9 elist) ":" (nth 6 elist) ":" (nth 7 elist)))
  ;; 	 (entry1 (gethash loc1 gklee-error-htable))
  ;; 	 (err-code (nth 1 elist)) ;try with "rrbc"
  ;; 	 (loc1Valid (and (> (length (nth 7 elist)) 0)
  ;; 			 (> (length (nth 6 elist)) 0)
  ;; 			 (> (length (nth 9 elist)) 0)
  ;; 			 (> (length (nth 8 elist)) 0)))
  ;; 	 )
  ;; 					;if there is an entry for the first location, check if the error is in its list
  ;;   (if entry0
  ;; 	(let ((err-list (gklee-get-error-list entry0 err-code)))
  ;; 	  (if (and err-list loc1Valid)
  ;; 	      (gklee-add-loc-to-error-list loc1 err-list)
  ;; 	    (if loc1Valid
  ;; 		(gklee-add-err-list entry0 err-code loc0 loc1))
  ;; 	    (gklee-add-err-list entry0 err-code loc0)))
  ;;     (if loc1Valid
  ;; 	  (puthash loc0 (list (list err-code loc0 loc1)) gklee-error-htable)
  ;; 	(puthash loc0 (list (list err-code loc0)) gklee-error-htable)
  ;; 	))
  ;;   (if (and entry1 loc1Valid)
  ;; 	(let ((err-list (gklee-get-error-list entry1 err-code)))
  ;; 	  (if err-list
  ;; 	      (gklee-add-loc-to-error-list loc0 err-list)
  ;; 	    (gklee-add-err-list entry1 err-code loc1 loc0))
  ;; 	  )
  ;;     (if loc1Valid
  ;; 	  (puthash loc1 (list (list err-code loc0 loc1)) gklee-error-htable))
  ;;     )
  ;;   )
  ;;)    

(defun gklee-process-record ()
  "This function parses the contents of the gklee-record (created by
the Gklee execution filter) to extract
Gklee data output"
  (let ((record gklee-record))
    (if (not (null record))
	(progn
	  (if (string-match "path num explored here: \\([0-9]+\\)" record)
	      (setq gklee-process-record-count (string-to-number (match-string 1 record))))
	  (if (string-match "emacs:\\([[:graph:][:space:]]*\\):\\([[:graph:][:space:]]*\\):\\([[:graph:][:space:]]*\\):\\([[:graph:][:space:]]*\\):\\([[:graph:][:space:]]*\\):\\([[:graph:][:space:]]*\\)" record)
	      (gklee-add-trace-error-info (split-string record ":"))
	    )
	  (if (string-match "threads \\([0-9]+\\) and \\([0-9]+\\) incur a \\(R-W\\|W-W\\) race on" record)
	      (setq gklee-race-info-list 
		    (append gklee-race-info-list (list (cons
							(+
							 gklee-process-record-count
							 1)
							(format
							 (concat
							  "Encountered a %s race condition between\n"
							  "threads %s and %s\n\n")
							 (match-string 3 record)
							 (match-string 1 record)
							 (match-string 2 record)))))))
	  (if (string-match "ASSERTION FAIL:\\([[:graph:][:space:]]*\\)$" record)
	      (setq gklee-assertion-info-list 
		    (append gklee-assertion-info-list (list (cons
							     (+ 
							      gklee-process-record-count
							      1)
							     (format
							      "Assertion Failure: %s\n"
							      (match-string 1 record)))))))
	  (if (or (string-match "GKLEE: Thread \\([0-9]+\\) and Thread \\([0-9]\\) encounter different barrier sequences" record)
		  (string-match "GKLEE: Thread \\([0-9]+\\) and Thread \\([0-9]\\) encounter the barrier sequences with different length" record)
		  )
	      (setq gklee-deadlock-info-list 
		    (append gklee-deadlock-info-list (list (cons
							    (+
							     gklee-process-record-count
							     1)
							    (format
							     (concat "Possible deadlock due to barrier mismatch\n"
								     "between threads %s and %s\n\n")
							     (match-string 1 record)
							     (match-string 2 record)))))))
	  
	  (if (string-match "Configuration: " record)
	      (setq gklee-config-info-list (append gklee-config-info-list 
						   (list (substring record (match-end 0)))))
	    )
	  (if (string-match (concat "BC:"
				    "\\([0-9]+\\):\\([0-9]+\\):\\([0-9]+\\)"
				    ":\\([0-9]+\\):\\([0-9]+\\):\\([0-9]+\\)"
				    ":\\([0-9]+\\)") record)
	      (setq gklee-bankcon-info-list 
		    (append gklee-bankcon-info-list (list (cons
							   (string-to-number (match-string 1 record))
							   (format 
							    (concat "%s%% warps bank conflicted\n"
								    "%s%%  BIs bank conflicted\n\n")
							    (match-string 2 record)
							    (match-string 5 record)))))))
	  (if (string-match (concat "MC:" 
				    "\\([0-9]+\\):\\([0-9]+\\):\\([0-9]+\\)"
				    ":\\([0-9]+\\):\\([0-9]+\\):\\([0-9]+\\)"
				    ":\\([0-9]+\\)") record)
	      (setq gklee-memcol-info-list 
		    (append gklee-memcol-info-list (list (cons
							  (string-to-number (match-string 1 record))
							  (format
							   (concat "%s%% warps memory coalesced\n"
								   "%s%% BIs memory coalesced\n\n")
							   (match-string 2 record)
							   (match-string 5 record)
							   ))))))
	  (if (string-match (concat "WD:"
				    "\\([0-9]+\\):\\([0-9]+\\):\\([0-9]+\\)"
				    ":\\([0-9]+\\):\\([0-9]+\\):\\([0-9]+\\)"
				    ":\\([0-9]+\\)") record)
	      (setq gklee-warpdiv-info-list (append gklee-warpdiv-info-list (list (cons
										   (string-to-number (match-string 1 record))
										   (format
										    (concat "%s%% warps warp divergent\n"
											    "%s%% BIs warp divergent\n\n")
										    (match-string 2 record)
										    (match-string 5 record)
										    ))))))
	  (if (string-match "KLEE: done: " record)
	      (setq gklee-summary-info-list 
		    (append gklee-summary-info-list (list (substring record (match-end 0))))))
	  (setq gklee-record nil)))))

(defun gklee-reset-run-state ()
  (setq gklee-instruction-count 0)
  (setq gklee-path-count 0)
  (setq gklee-test-count 0)
  (setq gklee-run-buffer buffer)
  (setq buffer-read-only nil)
  (setq gklee-tmp-dir (substring (shell-command-to-string "mktemp -d") 0 -1))
  (setq gklee-compile-process nil)
  (setq gklee-run-process nil)
  (setq gklee-record nil)
  (setq gklee-race-info-list nil)
  (setq gklee-assertion-info-list nil)
  (setq gklee-deadlock-info-list nil)
  (setq gklee-config-info-list nil)
  (setq gklee-bankcon-info-list nil) 
  (setq gklee-memcol-info-list nil)
  (setq gklee-warpdiv-info-list nil)
  (setq gklee-summary-info-list nil)
  (setq gklee-process-record-count 0)
  (setq gklee-trace-buffer nil)
  (setq gklee-trace-source-buffers nil)
  (setq gklee-trace-first-err-line nil)
  (setq gklee-trace-error-lines 0)
  (setq gklee-filter-obarray 
	(make-vector gklee-filter-obarray-size 0))
  (setq gklee-warp-count 0)
  ;;(clrhash gklee-error-htable)
  (setq gklee-error-alist nil)
  )

(defun gklee-run-filter (process output)
  "This function is called periodically by the Emacs process loop, passing 
stdout from the Gklee execution"
  (gklee-process-output((process-buffer process) output))
)

(defun gklee-process-output (buffer output)
  (with-current-buffer buffer
    (let ((output (concat gklee-unparsed output))
	  (buffer-read-only nil)
	  (si 0) 
	  ei)
      (while (setq ei (string-match "[\n]" output si))
	(setq gklee-record (substring output si (match-beginning 0)))
	(setq si (match-end 0))
	(gklee-insert-debug)
	(gklee-process-record)
	(gklee-partial-refresh)
	)
      (setq gklee-unparsed (substring output si))
      )))


;gklee-context-menu.el
;Guass Group, University of Utah
;January 19, 2012

;Author(s): Tyler Sorensen
;(Add your name if you work on it)

;This file creates the context menu that allows
;users to filter in and out lines by selecting them

;This file requires that the user already have the
;"gklee-mode.el file loaded and that this file is
;loaded while they are in their C file (in order 
;to locally bind the right click) (fixable?)

;Furthermore, this will fail if the file name has
;any "special" regex characters in it (the path is 
;fine) Perhaps that is fixable also? Also this doesn't
;handle swapping "All_Locations" text.

;This file will need to be updated if any of the in
;naming conventions in gklee-loc-syms is changed. 
;gklee-loc

;Ideas for future:
;Add a filter button to filter out all non-offending threads

(defun gklee-source-right-click (gklee-event)
  "This is the method that is bound to the right click button."
  (interactive "e")

  ;If the trace isn't generated display blacked out items
  (if (not gklee-trace-buffer)
      (x-popup-menu t 
      (list "Trace not Generated Yet"
      '("--" "Filter lines in" "Filter lines out"
      "Filter other lines in"  "Filter other lines out" )))

    (progn
      (let (choice max-line min-line)

	;If they have a region selected, take into account 
	;both locations
	(if (region-active-p)
	    (progn
	      (setq max-line 
              (max (line-number-at-pos (point)) (line-number-at-pos (mark))))

	      (setq min-line
              (min (line-number-at-pos (point)) (line-number-at-pos (mark))))

	      (setq choice 
              (x-popup-menu t 
	      (list (concat "Line: " (number-to-string min-line)
		    " - " (number-to-string max-line))
		    '("--" ("Filter lines in" .  1) ("Filter lines out" .  2)
		     ("Filter other lines in" . 3) ("Filter other lines out" . 4) "--" ("Start Stepping" . 5)))))

	;Find out which choice they made
	(if (equal choice 1)
	    (gklee-filter-lines min-line max-line t))
	(if (equal choice 2)
	  (gklee-filter-lines min-line max-line nil))
	(if (equal choice 3)
	    (gklee-filter-other-lines min-line max-line t))
	(if (equal choice 4)
	    (gklee-filter-other-lines min-line max-line nil))
	(if (equal choice 5)
	    (gklee-start-step))
)

	;else roughly the same thing but with one line
	(progn
	  (mouse-set-point gklee-event)
	  (setq max-line (line-number-at-pos (point)))
	  (setq choice 
          (x-popup-menu t 
	  (list (concat "Line: " (number-to-string max-line))
		'("--" ("Filter line in" .  1) ("Filter line out" .  2)
		 ("Filter other lines in" . 3) ("Filter other lines out" . 4) "--" ("Start Stepping" . 5)))))

	(if (equal choice 1)
	    (gklee-filter-lines max-line max-line t))
	(if (equal choice 2)
	  (gklee-filter-lines max-line max-line nil))
	(if (equal choice 3)
	    (gklee-filter-other-lines max-line max-line t))
	(if (equal choice 4)
	    (gklee-filter-other-lines max-line max-line nil))
	(if (equal choice 5)
	    (gklee-start-step))
	))))))


(defun gklee-filter-lines (min-line max-line in-out)
"Filters either in or out (based on IN-OUT) all the lines in the trace starting with MIN-LINE and ending with MAX-LINE"

 ;Lists are linked, this probably takes a long time, there may be a way to
 ;to optimize the loop
 ;in fact A LOT OF THINGS COULD BE OPTIMIZED (if speed becomes an issue, we could look at this)

 (let (
       (i 0) ; looping variable
       isym
       list-item
       regex)

   ;Iterate through the list...
   (while (< i (length gklee-loc-syms))

     (setq isym (nth i gklee-loc-syms)) ;Get the symbol
     (setq list-item (symbol-name isym)) ;Get the name of the symbol
     (setq i (+ 1 i))                   ;i++
     (setq regex (concat ".*" 
                         (file-name-sans-extension (file-name-nondirectory buffer-file-name))
                         "[^/]*:\\([[:digit:]]*\\)$"))
     
     ;Possible bugs because of regex expression .* (for path up to file name) FILENAME WITHOUT EXTENSION
     ;[^/] for no further directories then : digit to get the line number.
     ;         if filename (without path or extension) contains a special regex character (or '/' or ':' or '(' or ')')
     (if (string-match regex list-item)
	 (progn
	   ;Make sure it's in range
	   (if (and (<= (string-to-number (match-string 1 list-item)) max-line)
		    (>= (string-to-number (match-string 1 list-item)) min-line))
	       (progn
		 (gklee-filter-item-no-pos isym in-out))))))
   (redraw-display))) ;;redraw-display is a hack to get the buffers to update after filtering


(defun gklee-filter-other-lines (min-line max-line in-out)
"Filters either in or out (based on IN-OUT) all lines except the lines in the trace starting with MIN-LINE and ending with MAX-LINE"
;Pretty similiar to gklee-filter-lines. See that method for documentation (some of it is rather important)
 (let (
       (i 0)
       isym
       list-item
       regex)

   (while (< i (length gklee-loc-syms))
     (setq isym (nth i gklee-loc-syms))
     (setq list-item (symbol-name isym))
     (setq i (+ 1 i))
     (setq regex (concat ".*" 
                         (file-name-sans-extension (file-name-nondirectory buffer-file-name))
			 "[^/]*:\\([[:digit:]]*\\)$"))     

     (if (string-match regex list-item)
     	 (progn
	   (if (or (> (string-to-number (match-string 1 list-item)) max-line)
		   (< (string-to-number (match-string 1 list-item)) min-line))
	       (gklee-filter-item-no-pos isym in-out)))
       (progn
	 (if (not (equal list-item "All_Locations"))
	     (gklee-filter-item-no-pos isym in-out)))))
   
   (redraw-display))) ;;redraw-display is a hack to get the buffers to update after filtering



(defun gklee-filter-item-no-pos (isym io)
  "This function takes a symbol ISYM 
and performs the filter operation either in or out based on IO (shuffling filter
symbols around buffer-invisibility-spec's)"

(let* ((buf (buffer-name))
      (sym-name (symbol-name isym))
      )
  (gklee-add-to-invisibility-spec isym)
  (if (and
       (> (length sym-name) 2)
       (equal (substring (symbol-name isym) 0 3) "All"))
      (gklee-filter-all isym io)
    (if (equal (substring (symbol-name isym) 0 1) "W")
	(gklee-filter-warp isym io)))
  (if io
      (progn
	(with-current-buffer gklee-trace-buffer
	  (remove-from-invisibility-spec isym))
	(with-current-buffer gklee-available-filter-buffer
	  (remove-from-invisibility-spec isym))
	(with-current-buffer gklee-active-filter-buffer
	  (add-to-invisibility-spec isym))
	)
    (progn
    (with-current-buffer gklee-trace-buffer
      (gklee-add-to-invisibility-spec isym))
    (with-current-buffer gklee-active-filter-buffer
      (remove-from-invisibility-spec isym))
    (with-current-buffer gklee-available-filter-buffer
      (add-to-invisibility-spec isym)) 
    ))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Sets up windows
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun gklee-create-windows (top-l top-r bottom-l bottom-r)
  "This function closes all windows in a frame and opens up 4 new ones where the arguements are buffers that are to be placed in each window."
  (interactive)
  (delete-other-windows)

  (split-window-vertically)
  (split-window-horizontally)

  (get-buffer-create top-l)
  (switch-to-buffer top-l)
  (other-window 1)

  (get-buffer-create top-r)
  (switch-to-buffer top-r)
;  (set-window-dedicated-p (selected-window) t)
  (other-window 1)

  (get-buffer-create bottom-l)
  (switch-to-buffer bottom-l)
;  (set-window-dedicated-p (selected-window) t)
  (split-window-horizontally)
  (other-window 1)

  (get-buffer-create bottom-r)
  (switch-to-buffer bottom-r)
;  (set-window-dedicated-p (selected-window) t)
  (other-window -3)
  )


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Starting stepping code
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defvar gklee-step-trace-beg nil)
(defvar gklee-step-trace-begM nil)
(defvar gklee-sc-overlay-arrow-position)
(defvar gklee-trace-overlay-arrow-position)
(defvar gklee-step-trace-error nil)


;These didn't work before. Just as easy to change them
;in the actual code though as they each just appear once.
;(defvar gklee-hl-color "purple")
;(defvar gklee-step-key "n")
;(defvar gklee-quit-key "q")
;(defvar gklee-continue-key "q")


;gklee-step-mode. This mode is put on both the source and
;trace file. It just assigns hot keys for navigating the step
;features of gklee.
(define-minor-mode gklee-step-mode
  "Toggle Gklee Step Mode."
  :init-value nil
  :lighter " Step Mode"
  :keymap
  '(("n" . gklee-step)
    ("q" . gklee-step-quit-keypress)
    ("c" . gklee-step-contine-keypress))
)


(defun gklee-start-step ()
  "This function should be called with stepping mode is just started. It resets the windows to a good
layout for stepping and starts stepping at the begining of the trace. After this method call, use
gklee-step to perform a single step, gklee-continue-keypress to continue to the next error and gklee-quit-keypress
to quit the stepping mode"
  (interactive)
  (let (beg-line
	beg-mark
	(end-mark (make-marker))
	trace-line
	trace-lineP
	file-name
	(found nil)
	(end-buffer nil)
	)
    
    ;Set up the windows into a nice layout.
    (gklee-create-windows gklee-source-buffer gklee-trace-buffer gklee-available-filter-buffer gklee-active-filter-buffer)
    
    ;put trace window into gklee-step-mode
    (select-window (get-buffer-window gklee-trace-buffer))
    (if (not gklee-step-mode)
	(gklee-step-mode))
    
    ;goto begining of file and search for first real line
    ;and check for end of buffer.
    (goto-char (point-min))
    (while (and (not found) (not end-buffer))
      (setq beg-line (gklee-find-real-line (point)))
      (if (not beg-line)
	  (setq end-buffer t)
	(progn
	  (goto-char beg-line)
	  (setq beg-mark (point-marker))
	  (end-of-line)
	  (setq end-mark (point-marker))
	  (setq trace-line (buffer-substring-no-properties beg-mark end-mark))
	  (setq file-name (gklee-get-file-path trace-line))
	  (if file-name
	      (setq found t))
	  )))
    
    ;if end of buffer, quit step mode and report it.
    (if end-buffer
	(progn
	  (message "Trace File empty! Exiting step mode.")
	  (gklee-step-quit))
      (progn

	;check to see if's an error.
	(if (get-text-property beg-line 'face)
	    (setq gklee-step-trace-error t))
	
	;color line and set global variables.
	(gklee-color-trace-line beg-line)
	(setq gklee-step-trace-beg beg-line)	
	(setq gklee-step-trace-begM (make-marker))
	(set-marker gklee-step-trace-begM gklee-step-trace-beg)
	
	(add-to-list 'overlay-arrow-variable-list 'gklee-sc-overlay-arrow-position)
	(add-to-list 'overlay-arrow-variable-list 'gklee-trace-overlay-arrow-position)
	(gklee-goto-line gklee-step-trace-beg)
	
	;clean up markers
	(set-marker beg-mark nil)
	(set-marker end-mark nil)
	(message "Gklee step-mode. Press q to exit")
	))))

(defun gklee-get-file-path (trace-line)
(if (string-match ".*File \\([[:ascii:]]*\\)" trace-line)
(match-string 1 trace-line)
nil)
)

(defun gklee-color-trace-line (beg-line)
  "This function colors the line begining with position LINEBEG
purple."
  (let ((beg beg-line)
	end)
    (save-excursion
      (select-window (get-buffer-window gklee-trace-buffer))
      (goto-char beg)
      (end-of-line)
      (setq end (point))
      (toggle-read-only)
      (add-text-properties beg end (list 'face (list :foreground "purple")))
      (toggle-read-only))
    ))

(defun gklee-uncolor-trace-line (beg-line)
  "This function uncolors the line in the trace file starting with
beg-line. If the line was an error line before, it colors it red again"
  (let ((beg beg-line)
	end)
    (save-excursion
      (select-window (get-buffer-window gklee-trace-buffer))
      (goto-char beg)
      (end-of-line)
      (setq end (point))
      (toggle-read-only)
      
					;check to see if it's an error
      (if gklee-step-trace-error
	  (add-text-properties beg end (list 'face (list :foreground "red")))
	(remove-text-properties beg end '(face nil)))
      
					;reset error variable.
      (setq gklee-step-trace-error nil)
      (toggle-read-only)
      )))

(defun gklee-find-real-line (beg-line)
  "This function finds the next line in the trace file
that isn't invisible or an LLVM instruction. It starts 
searching at LINEBEG. If it reaches the end of the buffer,
NIL is returned."
  (let ((curPoint beg-line)
    (beg-mark (make-marker))
    (end-mark (make-marker))
    (eof nil)
    trace-line
    line-watch)
    (set-marker beg-mark curPoint)
    (end-of-line)
    (set-marker end-mark beg-line)
    (setq trace-line (buffer-substring beg-mark end-mark))
    
    (while (and (or (invisible-p beg-mark) (equal (length trace-line) 0)) (not eof))
      (setq line-watch (forward-line))
      (setq curPoint (point))
      (set-marker beg-mark curPoint)
      (end-of-line)
      (set-marker end-mark (point))
      (setq trace-line (buffer-substring beg-mark end-mark))
      (if (equal line-watch 1)
	  (setq eof t))
      )
    (set-marker end-mark nil)
    (if (equal line-watch 1)
	nil
      (marker-position beg-mark))
    ))

(defun gklee-step ()
  "This function should only be called after gklee-start-step. It performs the action of a 
single step through the trace file. It is very similiar to start step but with just enough
differences to warrent it's own function (uncoloring lines updating variables etc)"
  (interactive)
  (let (beg-line
	beg-mark
	(end-mark (make-marker))
	trace-line
	file-name
	(found nil)
	(end-buffer nil)
	)
    
    (if (not (get-buffer-window gklee-trace-buffer))
	(switch-to-buffer gklee-trace-buffer))
    (select-window (get-buffer-window gklee-trace-buffer))
    (goto-char gklee-step-trace-beg)
    (gklee-uncolor-trace-line (point))
    (forward-line)
    
    (while (and (not found) (not end-buffer))
      (setq beg-line (gklee-find-real-line (point)))
      (if (not beg-line)
	  (setq end-buffer t)
	(progn
	  (goto-char beg-line)
	  (setq beg-mark (point-marker))
	  (end-of-line)
	  (setq end-mark (point-marker))
	  (setq trace-line (buffer-substring-no-properties beg-mark end-mark))

	  (setq file-name (gklee-get-file-path trace-line))
	  (if file-name
	      (setq found t))
	  )))
    
    (if end-buffer
	(progn
	  (message "End of trace buffer reached, exiting step mode")
	  (gklee-step-quit))
      (progn
	(if (get-text-property beg-line 'face)
	    (setq gklee-step-trace-error t))
	
	(gklee-color-trace-line beg-line)
	(setq gklee-step-trace-beg beg-line)
	(set-marker gklee-step-trace-begM gklee-step-trace-beg)
	(gklee-goto-line gklee-step-trace-beg)
		
	(set-marker beg-mark nil)
	(set-marker end-mark nil)
	
	(message "Gklee step-mode. Press q to exit")
	))))

(defun gklee-step-quit ()
  "General step quit function. This function makes all the overlay arrows point to nil
and removes all buffers from the gklee-step minor mode"
  (loop for buf being the buffers do
	(with-current-buffer buf
	  (if gklee-step-mode
	      (gklee-step-mode 0))))
  (set-marker gklee-step-trace-begM nil)
  (delq 'gklee-trace-overlay-arrow-position overlay-arrow-variable-list)
  (delq 'gklee-sc-overlay-arrow-position overlay-arrow-variable-list)
  (setq overlay-arrow-position nil)
  )

(defun gklee-step-quit-keypress ()
  "This function is called with the quit key is pressed in the gklee-step minor mode
It uncolors the trace line and calles the regular gklee-step-quit function"
  (interactive)
  
					;Uncolor the line
  (select-window (get-buffer-window gklee-trace-buffer))
  (goto-char gklee-step-trace-beg)
  (gklee-uncolor-trace-line (point))
  
					;Do regular quit stuff (overlay errors, get out of modes etc)
  (gklee-step-quit)
  (message "Step mode has exited")
  )

(defun gklee-step-contine-keypress ()
  "This function is called when the continue key is presssed in the gklee-step minor mode.
It scans through the trace and stops at the closest error line"
  (interactive)
  (let (beg-line
	beg-mark
	(end-mark (make-marker))
	trace-line
	file-name
	(found nil)
	(end-buffer nil)
	)
    
					;Pull up a the trace buffer
    (if (not (get-buffer-window gklee-trace-buffer))
	(switch-to-buffer gklee-trace-buffer))
    (select-window (get-buffer-window gklee-trace-buffer))
    (goto-char gklee-step-trace-beg)
    
					;uncolor the line and move forward
    (gklee-uncolor-trace-line (point))
    (forward-line)
    
					;while an error line isn't found and it's not the end of a buffer,
					;go to next lines.
    (while (and (not found) (not end-buffer))
      (setq beg-line (gklee-find-real-line (point)))
      (if (not beg-line)
	  (setq end-buffer t)
	(progn
	  (goto-char beg-line)
	  (setq beg-mark (point-marker))
	  (end-of-line)
	  (setq end-mark (point-marker))
	  (setq trace-line (buffer-substring-no-properties beg-mark end-mark))
	  (setq file-name (gklee-get-file-path trace-line))
	  (if (and file-name (get-text-property beg-line 'face))
	      (progn
		(setq found t)
		(setq gklee-step-trace-error t)))
	  )))
    
    ;If it's the end of the buffer, exit.
    (if end-buffer
	(progn
	  (message "End of trace buffer reached, exiting step mode")
	  (gklee-step-quit))
      (progn 
	
	;color the line
	(gklee-color-trace-line beg-line)
	
	;set global variables
	(setq gklee-step-trace-beg beg-line)
	(set-marker gklee-step-trace-begM gklee-step-trace-beg)
	
	;go to the source file
	(gklee-goto-line gklee-step-trace-beg)  

	
	;clean up markers
	(set-marker beg-mark nil)
	(set-marker end-mark nil)
	
	(message "First Error line since last step")
	))))
