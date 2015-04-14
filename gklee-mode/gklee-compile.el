;;; -*- lexical-binding: t -*-
(load "gklee-command-args.el")
(defconst gklee-compile-buffer-name "*gklee-compile*")
(defconst gklee-run-buffer-name "*gklee-run*")
(defconst gklee-results-buffer-name "*gklee-results*")
(defconst gklee-source-buffer-name "")

(defconst gklee-compiler "gklee-nvcc") ;; must be in user's PATH
(defconst gklee-runner "gklee") ;; must be in user's PATH
(defvar gklee-process nil) ;; the currently running gklee async process
(defvar gklee-file-name "") ;; current source file

(defvar gklee-json-races nil)
(defvar gklee-last-button nil)

(require 'json)

(defun gklee-compile ()
  (interactive)
  (gklee-kill)
  (let ((file-name (buffer-file-name)) ;; get the filename to compile
	(buffer-name (buffer-name)) ;; and buffer name (may be different)
	(process-connection-type nil) ;; use a pipe to connect to gklee-nvcc
	(output-buffer (get-buffer-create gklee-compile-buffer-name)))
    (if file-name
	(with-current-buffer output-buffer
	  (setq buffer-read-only nil)
	  (erase-buffer)
	  (buffer-disable-undo)
	  ;; set globals for use later
	  (setq gklee-file-name file-name) 
	  (setq gklee-source-buffer-name buffer-name)
	  (insert "executing with:\n")
	  (insert (concat "\t" gklee-compiler " " 
			  (gklee-get-compile-command-args file-name)))
	  (insert "\n\nin directory:\n")
	  (insert (concat "\t" (shell-command-to-string "pwd")))
	  (insert "\n\ncommand output:\n")
	  ;; bind the process so we can use it if it needs killing
	  (let ((process (apply 'start-process
				gklee-compile-buffer-name
				gklee-compile-buffer-name
				(list gklee-compiler
				      (gklee-get-compile-command-args file-name)))))
	    (setq gklee-process process)
	    (if (not (called-interactively-p))
		process ;; just go if called from another function
	      (set-process-sentinel process (lambda (p e)
					      (with-current-buffer gklee-compile-buffer-name
						(insert "\nDone")
						(setq buffer-read-only t))))
	      (split-window-horizontally)
	      (other-window 1)
	      (switch-to-buffer output-buffer)
	      process)))
      (message "File does not exist"))))



(defun gklee-toggle-lines (button line-a line-b buffer)
  ;; highlight button
  (with-current-buffer gklee-results-buffer-name
    (if gklee-last-button
	(add-text-properties (button-start gklee-last-button) (button-end gklee-last-button)
			 '(face (:background nil))))
    (add-text-properties (button-start button) (button-end button)
			 '(face (:background "green"))))
  ;; highlight source lines
  (with-current-buffer buffer
    (if (equal button gklee-last-button)
	(progn (normal-mode)
	        (with-current-buffer gklee-results-buffer-name
		  (if gklee-last-button
		      (add-text-properties (button-start gklee-last-button) 
					   (button-end gklee-last-button)
					   '(face (:background nil)))))
	       (setq gklee-last-button nil))
      (fundamental-mode)
      (remove-text-properties (point-min) (point-max) '(face nil))
      (if (equal line-a line-b)
	  (progn (goto-char (point-min)) (forward-line (1- line-a))
		 (add-text-properties (point) (progn (forward-line 1) (point))
				      '(face (:foreground "purple"))))
	(goto-char (point-min)) (forward-line (1- line-a))
	(add-text-properties (point) (progn (forward-line 1) (point))
			     '(face (:foreground "red")))
	(goto-char (point-min)) (forward-line (1- line-b))
	(add-text-properties (point) (progn (forward-line 1) (point))
			     '(face (:foreground "blue"))))
      (setq gklee-last-button button))))

(defun gklee-display-races ()
  (delete-other-windows)
  (split-window-horizontally)
  (get-buffer-create gklee-source-buffer-name)
  (switch-to-buffer gklee-source-buffer-name)
  (other-window 1)
  (get-buffer-create gklee-results-buffer-name)
  (switch-to-buffer gklee-results-buffer-name)
  (with-current-buffer gklee-results-buffer-name
    (erase-buffer)
    (let ((i 0)
	  race-alist)
      (dolist (race gklee-json-races)
	(setq race-alist (json-read-from-string race))
	(lexical-let
	 ((line-a (string-to-number (cdr (assoc-string "source-line-a" race-alist))))
	  (line-b (string-to-number (cdr (assoc-string "source-line-b" race-alist)))))
	 (insert (propertize (concat "Race " (number-to-string i) " ")
			     'face '((foreground-color . "green"))))
	(insert (propertize (concat (cdr (assoc-string "race-type" race-alist))
				    " ")
			    'face '((foreground-color . "green"))))
	 (insert-button "Show Race"
	   'action #'(lambda (x) (gklee-toggle-lines x line-a line-b gklee-source-buffer-name))
	   'follow-link t)
	(insert (propertize (concat "\n\tThread id a: "
				    (cdr (assoc-string "thread-id:" race-alist)))
			    'face '((foreground-color . "red"))))
	(insert (propertize (concat "\n\tThread id b: "
				    (cdr (assoc-string "thread-id:" race-alist)))
			    'face '((foreground-color . "blue"))))
	(insert "\n\n"))
	(setq i (+ i 1))))
    ))


(defun gklee-filter-buffer (process event)
  (setq gklee-json-races nil)
  (if (equal 'exit (process-status process))
      (if (not (equal 0 (process-exit-status process)))
	  (message (concat "gklee: " gklee-runner " running failed\n"))
	;; Filter content
	(with-current-buffer gklee-run-buffer-name
	  (let ((output (buffer-substring-no-properties 1 (point-max)))
		(start-of-next-line 0)
		end-of-line
		current-line
		(in-race nil)
		(emacs-line nil))
	    (while (setq end-of-line
			 (string-match "[\n]" output start-of-next-line))
	      (setq current-line
		    (substring output start-of-next-line (match-beginning 0)))
	      (setq start-of-next-line (match-end 0))
	      (setq emacs-line (string-match-p (regexp-quote"[GKLEE]: emacs:") current-line))
	      (if (and in-race (not emacs-line))
		  (setq in-race nil))
	      (if emacs-line
		  (if in-race
		      ;; append the string to the first element
		      (setcar gklee-json-races
			      (concat (car gklee-json-races) (substring current-line 15 nil)))
		    ;; add the string as an new element at the front
		    (setq in-race t)
		    (setq gklee-json-races (cons (substring current-line 15 nil) gklee-json-races)))))))
	(setq gklee-json-races (reverse gklee-json-races))
	(gklee-display-races))
    (message (concat "gklee: " gklee-runner " running failed"))))

(defun gklee-run-sentinel (process event)
  (if (equal 'exit (process-status process))
      (if (not (equal 0 (process-exit-status process)))
	  (message (concat "gklee: " gklee-compile " compilation failed\n"))
	;; Finish up with compile buffer
	(with-current-buffer gklee-compile-buffer-name
	  (insert "\nDone.\n")
	  (setq buffer-read-only t))
	;; Run gklee itself
	(let ((process-connection-type nil) ;; use a pipe to connect to gklee
	      (buffer (get-buffer-create gklee-run-buffer-name))) ;; output buffer
	  (with-current-buffer buffer
	    (setq buffer-read-only nil)
	    (erase-buffer)
	    (buffer-disable-undo)
	    (insert "running with: \n")
	    (insert (concat  "\t" gklee-runner " "
			     (gklee-get-gklee-command-args
			      (substring gklee-file-name 0 -3))))
	    (let ((process (apply 'start-process
				  gklee-run-buffer-name
				  gklee-run-buffer-name
				  (list gklee-runner
					(gklee-get-gklee-command-args
					 (substring gklee-file-name 0 -3))))))
	      (setq gklee-process process)
	      (set-process-sentinel process 'gklee-filter-buffer)
	      process))))))


(defun gklee-run ()
  (interactive)
  (set-process-sentinel (gklee-compile) 'gklee-run-sentinel))


(defun gklee-kill ()
  "Kills the curerntly running Gklee process
If called interactively prompts the user first"
  (interactive)
  (if gklee-process
      (if (or (not (called-interactively))
	      (y-or-no-p "Do you want to kill Gklee?")
	      (delete-process gklee-process))))
  (if (and (not gklee-process)
	   (called-interactively-p))
      (message "Gklee is stopped")))


