;;; -*- lexical-binding: t -*-
;; Adapted by Ian Briggs from gklee-mode.el
;; ianbriggs@utah.edu

(defgroup gklee-mode nil
  "Utilizes Gklee to analyse cuda code"
  :tag "Gklee-Mode"
  :group 'external)

(defgroup gklee-run-options nil
  "Gklee command line options"
  :tag "Gklee run options"
  :group 'gklee-mode)

(defgroup gklee-compile-options nil
  "Gklee compiler command line options"
  :tag "Gklee compile options"
  :group 'gklee-mode)

(defgroup gklee-program-options nil
  "Program under test command line options"
  :tag "Gklee program under test options"
  :group 'gklee-mode)

;;****************************************************************************;;
;;**                            Option vars                                 **;;
;;****************************************************************************;;

;; Boolean flags
(defcustom gklee-verbose nil "Increases verbosity when set"
  :group 'gklee-run-options :tag "verbose" :type 'boolean)

(defcustom gklee-PR-use-dep nil "Uses dependecies for path reduction when set"
  :group 'gklee-run-options :tag "path reduction using dependencies" :type 'boolean)

(defcustom gklee-asm-verbose nil "Increases verbosity in asembly output when set"
  :group 'gklee-run-options :tag "asm verbosity" :type 'boolean)

(defcustom gklee-avoid-oob-check nil "Avoids out of bound check when set"
  :group 'gklee-run-options :tag "avoid oob check" :type 'boolean)

(defcustom gklee-bc-cov nil "Calculates bytecode coverage when set"
  :group 'gklee-run-options :tag "bytecode coverage" :type 'boolean)

(defcustom gklee-check-barrier-redundant nil "Checks for unneeded barriers when set"
  :group 'gklee-run-options :tag "check redundant barriers" :type 'boolean)

(defcustom gklee-check-div-zero nil "Checks for division by zero when set"
  :group 'gklee-run-options :tag "check divide by zero" :type 'boolean)

(defcustom gklee-check-volatile nil "Check for missing volatile declarations when set"
  :group 'gklee-run-options :tag "check missed volatile declarations" :type 'boolean)

(defcustom gklee-emit-all-errors nil "Generates test cases for all errors when set"
  :group 'gklee-run-options :tag "Generate all error tests" :type 'boolean)

(defcustom gklee-ignore-concur-bug nil "Continues upon a concurancy bug when set"
  :group 'gklee-run-options :tag "continue after concurancy bug" :type 'boolean)

(defcustom gklee-race-prune nil "Prunes paths not leading to races when set"
  :group 'gklee-run-options :tag "prune non-racing paths" :type 'boolean)

(defcustom gklee-simd-schedule nil "Uses SIMD aware scheduling when set"
  :group 'gklee-run-options :tag "SIMD scheduling" :type 'boolean)

(defcustom gklee-suppress-external-warnings nil "Supresses external warning when set"
  :group 'gklee-run-options :tag "supress external warnings" :type 'boolean)

(defcustom gklee-symbolic-config nil "Run Gkleep when set"
  :group 'gklee-run-options :tag "Use Gkleep" :type 'boolean)

;; integer options
(defcustom gklee-check-level -1 "the level of race checking used by Gklee, -1=unset"
  :group 'gklee-run-options :tag "check level" 
  :type '(integer :match (lambda (w val) (>= val -1))))

(defcustom gklee-device-capability -1 "device capability (-1)=Gklee defaiult (0)=1.0 and 1.1 (1)=1.2 and 1.3 (2)=2.x"
  :group 'gklee-run-options :tag "device capatbility" 
  :type '(integer :match (lambda (w val) (and (>= val -1) (<= val 2)))))

(defcustom gklee-max-memory -1 "memory limit on Gklee (0=no limit, -1=unset)"
  :group 'gklee-run-options :tag "memory limit" 
  :type '(integer :match (lambda (w val) (>= val -1))))

(defcustom gklee-max-time -1 "time limit on Gklee (0=no limit, -1=unset)"
  :group 'gklee-run-options :tag "time limit" 
  :type '(integer :match (lambda (w val) (>= val -1))))

;; strings
(defcustom gklee-gklee-user-args ""
  "Command line arguments to pass to Gklee"
  :group 'gklee-run-options :tag "user supplied arguments"
  :type 'string)

(defcustom gklee-compile-user-args ""
  "Command line arguments to pass to llvm"
  :group 'gklee-compile-options :tag "user supplied arguments"
  :type 'string)

(defcustom gklee-program-user-args ""
  "Command line arguments to pass to your program"
  :group 'gklee-program-args :tag "user supplied arguments"
  :type 'string)


;;****************************************************************************;;
;;**                       Command generatorss                              **;;
;;****************************************************************************;;
(defun gklee-get-compile-command-args (file-name)
  "Returns the command args for compiling the given filewith the current settings"
  (concat gklee-compile-user-args " " file-name))

(defun gklee-get-gklee-command-args (file-name)
  "Returns the command agrs for running Gklee on the given file for the current settings"
  (let ((bool-arguments  '((gklee-verbose . "-verbose")
			   (gklee-PR-use-dep . "-PF-use-dep")
			   (gklee-asm-verbose . "-asm-verbose")
			   (gklee-avoid-oob-check . "-avoid-oob-check")
			   (gklee-bc-cov . "-bc-cov")
			   (gklee-check-barrier-redundant . "-check-barrier-redundant")
			   (gklee-check-div-zero . "-check-div-zero")
			   (gklee-check-volatile . "-check-volatile")
			   (gklee-emit-all-errors . "-emit-all-errors")
			   (gklee-ignore-concur-bug . "-ignore-concur-bug")
			   (gklee-race-prune . "-race-prune")
			   (gklee-simd-schedule . "-simd-schedule")
			   (gklee-suppress-external-warnings . "-suppress-external-warnings")
			   (gklee-symbolic-config . "-symbolic-config")))

	(int-arguments '((gklee-check-level . "-check-level=")
			 (gklee-device-capability . "-device-compatibility=")
			 (gklee-max-memory . "-max-memory=")
			 (gklee-max-time . "-max-time="))))
    (concat "-emacs"
	    (mapconcat (lambda (elm) (if (symbol-value (car elm))
					 (concat " " (cdr elm))
				       ""))
		       bool-arguments "")
	    (mapconcat (lambda (elm) (if (>= (symbol-value (car elm)) 0)
					(concat " " (cdr elm) (symbol-value (car elm)))
				      ""))
		      int-arguments "")
	    " " gklee-gklee-user-args
	    " " file-name)))


;;****************************************************************************;;
;;**                       Global key bindings                              **;;
;;****************************************************************************;;

;; ;;TODO bind these
;; (global-set-key "\M-gor"  'gklee-open-remote-package)
(global-set-key "\M-gr"   'gklee-run)
(global-set-key "\M-gk"   'gklee-kill)
