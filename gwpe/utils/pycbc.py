from pycbc.transforms import apply_transforms
from pycbc.io import record

# add custom mass constraint to be read in by read_constraints_from_config
from pycbc.distributions import constraints

class MassConstraint:
    """Custom pycbc.distributions.constraints.Constraint object that evaluates
    to True if mass parameters (mass1 and mass2) obey the conventional notation
    where mass1 >= mass2.
    """
    name = "mass"

    def __init__(self, constraint_arg=None, transforms=None, **kwargs):
        self.constraint_arg = constraint_arg
        self.transforms = transforms
        for kwarg in kwargs.keys():
            setattr(self, kwarg, kwargs[kwarg])

    def __call__(self, params):
        """Evaluates constraint.
        """
        # cast to FieldArray
        if isinstance(params, dict):
            params = record.FieldArray.from_kwargs(**params)
        elif not isinstance(params, record.FieldArray):
            raise ValueError("params must be dict or FieldArray instance")

        # try to evaluate; this will assume that all of the needed parameters
        # for the constraint exists in params
        try:
            out = self._constraint(params)
        except NameError:
            # one or more needed parameters don't exist; try applying the transforms
            params = apply_transforms(params, self.transforms) if self.transforms else params
            out = self._constraint(params)
        if isinstance(out, record.FieldArray):
            out = out.item() if params.size == 1 else out
        return out

    def _constraint(self, params):
        """Evaluates constraint function.
        
        Warning: Requires priors to be specified as mass_1 and mass_2
                 in the PyCBC ini config file.
        """
        return params['mass_1'] >= params['mass_2']

constraints.constraints['mass'] = MassConstraint