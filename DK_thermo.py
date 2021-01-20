"""Pilot codebase to implement basic thermodynamic analyses, such as fixing states of substances

Humid air
- depends on psychrolib, is based on the ASHRAE Handbook

See also
- steam tables in Python: https://pythonhosted.org/thermopy/iapws.html

Author:     David Kayanan
Created:    Mar 25, 2020
Version:    early
"""

import psychrolib as psy
psy.SetUnitSystem(psy.SI)
print('psychrolib initialized to SI units')

import numpy as np
import warnings

# CONSTANTS
c_ = {
	'Patm': 101325      # atmospheric pressure in Pa

}
water = {
	'cp':  4184,                # J/kg-K
	'cp vapor 275 K': 1859,     # J/kg-K
	'cp vapor 375 K': 1890,     # J/kg-K
	'h vap 0°C': 2501*1000,     # J/kg
}


class humidair():
	"""Implements humid air (substance), which wraps some functions of psychrolib
	(https://github.com/psychrometrics/psychrolib)"""
	def __init__(self, P=c_['Patm']):
		# Temp
		self.TDryBulb  = None           # Dry Bulb Temp [C]
		self.TWetBulb = None            # Wet Bulb Temp [C]
		self.TDewPoint = None

		# Pressure
		self.P = P                  # Pressure [Pa]
		self.VapPres = None         # Partial pressure of water vapor in moist air [Pa]

		# Humidity
		self.HumRatio = None            # [kg water vapor / kg dry air]
		self.RelHum = None              # [0, 1]
		self.DegreeofSaturation = None  # [0, 1]

		# Enthalpy
		self.h_moist = None             # enthalpy [J/kg da] of humid air
		self.h_dry = None               # enthalpy [J/kg da] of dry air

		# Volume
		self.MoistAirVolume = None      # specific volume [m3/kg]

		return


	@staticmethod
	def fixstatefr_Tdb_RH_P(TDryBulb, RelHum, P=c_['Patm']):
		"""Fixes the state (i.e. calculates all state variables) from: Dry Bulb Temp, Relative Humidty and Pressure.
		Returns the humidair instance."""

		# 1) Init and set given
		air = humidair(P=P)
		air.TDryBulb = TDryBulb
		air.RelHum = RelHum

		# 2) Complete the psychrometrics

		# Humidity ratio in lb_H₂O lb_Air⁻¹ [IP] or kg_H₂O kg_Air⁻¹ [SI]
		# Wet-bulb temperature in °F [IP] or °C [SI]
		# Dew-point temperature in °F [IP] or °C [SI].
		# Partial pressure of water vapor in moist air in Psi [IP] or Pa [SI]
		# Moist air enthalpy in Btu lb⁻¹ [IP] or J kg⁻¹ [SI]
		# Specific volume of moist air in ft³ lb⁻¹ [IP] or in m³ kg⁻¹ [SI]
		# Degree of saturation [unitless]
		air.HumRatio, air.TWetBulb, air.TDewPoint, air.VapPres, air.h_moist, air.MoistAirVolume, \
		air.DegreeofSaturation = psy.CalcPsychrometricsFromRelHum(air.TDryBulb, air.RelHum, air.P)

		# 3) Get the dry air enthalpy
		air.h_dry = psy.GetDryAirEnthalpy(air.TDryBulb)

		return air

	@staticmethod
	def sensible_latent_heat_split(air_i, air_f):
		"""Estimates the split of sensible and latent heat absorbed by the air as it transitions from state air_i to
		air_f.

		PARAMETERS:
			air_i, air_f    Initial and final states of air (humidair instances)

		RETURNS:
			sensible            Estimated sensible heat as J/kg da
			latent              Estimated latent heat as J/kg da

			%error of LH        There is a slight error introduced by not considering at which temperature each
								incremental humidification occurs. This is returned as a percentage, and a warning is
								issued if it goes beyond 2%.

		METHOD:
			This is done by estimating the latent heat absorbed, and then deducting this from the exact enthalpy
			change to get the sensible heat. This uses the ideal gas expressions of the enthalpy, which is valid at
			Patm (recall: the higher the pressure, the more the gas deviates from ideal behavior).

			Note that psychrolib enthalpies are not referenced to 0C air and 0C liquid water.
			(humidair.fixstatefr_Tdb_RH_P(0, 0).h_moist is not zero)

		"""
		# ........................................................ 1) Setup utils
		c_pv = water['cp vapor 275 K']          # J/kg-K
		h_vap = water['h vap 0°C']              # J/kg

		LH_coeff = lambda T: c_pv*T + h_vap     # J/kg
		# Δh_LH = LH_coeff * Δw

		# ........................................................ 2) Get state vars
		w_i = air_i.HumRatio
		w_f = air_f.HumRatio

		T_i = air_i.TDryBulb
		T_f = air_f.TDryBulb

		h_i = air_i.h_moist
		h_f = air_f.h_moist

		# ........................................................ 3) Error due to unknown temp of humidification
		LH_coeff_err = (LH_coeff(T_f) - LH_coeff(T_i))/LH_coeff(T_i)

		if LH_coeff_err > 0.02:
			warnings.warn('The large temperature difference causes a relative error of {:0.1f}% in the latent heat '
			              'calculation'.format(LH_coeff_err*100))

		# ........................................................ 4) Calc LH, SH
		deltah_LH = LH_coeff(0.5*(T_f+T_i)) * (w_f-w_i)     # J/kg da
		deltah_SH = h_f-h_i - deltah_LH

		return deltah_SH, deltah_LH, LH_coeff_err


	def __repr__(self):
		return "Humid air at {:0.2f} °C, {:0.3f} RH".format(self.TDryBulb, self.RelHum)


	def viewstate(self):
		print('\n'.join(('Pressure: \t\t{} Pa\n'.format(self.P),
		                  'TDryBulb: \t\t{:.1f} °C'.format(self.TDryBulb),
		                  'TWetBulb: \t\t{:.1f} °C'.format(self.TWetBulb),
		                  'TDewPoint: \t\t{:.1f} °C\n'.format(self.TDewPoint),
		                  'Rel Humidity: \t\t{}'.format(self.RelHum),
		                  'Humidity Ratio: \t{:.4f} kg/kg\n'.format(self.HumRatio),
		                  'Enthalpy (total): \t{:.0f} J/kg d.a.'.format(self.h_moist),
		                  'Enthalpy (d.a.): \t{:.0f} J/kg d.a.\n'.format(self.h_dry),
		                  'Specific volume: \t{:0.2f} m3/kg'.format(self.MoistAirVolume),
		                  )))
		return


	def GetTWetBulbFromRelHum(self):
		self.TWetBulb = psy.GetTWetBulbFromRelHum(self.TDryBulb, self.RelHum, self.P)
		return

	def GetMoistAirEnthalpy(self):
		self.h_moist = psy.GetMoistAirEnthalpy(self.TDryBulb, self.HumRatio)


def liqwater_h(T, To=0):
	"""Calculates the specific enthalpy [J/kg] of liquid water at atmospheric pressure, for a given temperature (
	Celsius).

	Notes: This is calculated via the specific heat capacity of water at Patm, for a specified reference temp (O
	C by defualt). Some sources report slightly different values for the heat capacity, but we follow the definition
	of the calorie."""

	return  water['cp'] * (T-To)


def getstate(state, airseq):
	"Extracts the specified state from the sequence of humid air objects as a numpy array"
	return np.fromiter((getattr(_air, state) for _air in airseq), dtype='f8')



