dnl  Contains m4 macros for replicating
dnl  Fortran routines and interfaces, for types
dnl  integer, real, and real(r8), and array
dnl  dimensions up to rank 4.
dnl
dnl  Replicator concept inspired by Caesar package
dnl  LA-UR-00-5568, LA-CC-06-027
dnl  http://www.lanl.gov/Caesar/Caesar.html
dnl
dnl
define(`REPLICATE_INTERFACE_DIM', `dnl
    module procedure $1_$2x1
    module procedure $1_$2x2
    module procedure $1_$2x3
    module procedure $1_$2x4')dnl
define(`REPLICATE_INTERFACE_TYPE_DIM', `dnl
REPLICATE_INTERFACE_DIM(`$1', `i4')

REPLICATE_INTERFACE_DIM(`$1', `r4')

REPLICATE_INTERFACE_DIM(`$1', `r8')')dnl
dnl
dnl
define(`REPLICATE_ROUTINE_DIM', `dnl
ROUTINE_INSTANCE(`$1x1', `$2', `(:)')

ROUTINE_INSTANCE(`$1x2', `$2', `(:,:)')

ROUTINE_INSTANCE(`$1x3', `$2', `(:,:,:)')

ROUTINE_INSTANCE(`$1x4', `$2', `(:,:,:,:)')')dnl
dnl
define(`REPLICATE_ROUTINE_TYPE_DIM', `dnl
REPLICATE_ROUTINE_DIM(`i4', `integer')

REPLICATE_ROUTINE_DIM(`r4', `real')

REPLICATE_ROUTINE_DIM(`r8', `real(r8)')')dnl
