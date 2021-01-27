"""
Cooling towers basic modelling


Key Dependencies:
	- DK_thermo     humid air state (psychrometry via PsychroLib: https://psychrometrics.github.io/psychrolib/api_docs.html)
					liquid water enthalpy calculation

	- Pint 0.9      explicit units



Author:     D. Kayanan
Created:    Mar 30, 2020
Version:    Early
"""
# Python
from os import path

# Scipy ecosystem
import pandas as pd
import matplotlib.pyplot as plt

# More 3rd party
from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity

# DK
from scripts.DK_thermo import *

# from DK_Collections import basic_plot_polishing

# Constants from DK_thermo
cp_water = Q_(water['cp'], 'J/(kg*delta_degC)')

# Other constants
PathProj = path.dirname(path.abspath(__file__))
PathPlots = path.join(PathProj, 'Results', 'Plots')


def read_CTcatalog(filename, directory=None):
    """Reads the CT design file (pickled DataFrame) given by filename at the location directory. The directory
    defaults to the 'CT design' project folder. Returns the design DataFrame."""
    if directory is None:
        directory = path.join(PathProj, 'CT designs')

    CT_catalog = pd.read_pickle(path.join(directory, filename))

    # Check minimal columns
    assert set(CT_catalog.columns) >= {'Capacity [kW]', 'HWT [°C]', 'CWT [°C]', 'WBT [°C]', 'approach [C°]',
                                       'range [C°]', 'water flow [kg/s]', 'air flow [kg/s]', 'L/G',
                                       'evap loss [%]', 'Fan diameter [m]', 'CT perf slope', 'CT perf y-int',
                                       'Min Range [C°]', 'Max per unit water flow', 'Min per unit water flow'}, \
        "Columns are lacking"

    # parse_BldgToCTs require the CT catalog to be indexed by the kW values. This can definitely be changed in the
    # future.
    CT_catalog.index = pd.Index(CT_catalog['Capacity [kW]'].values, name='CT')

    return CT_catalog


def set_ambient(TDryBulb, RelHumidity, Pressure=c_['Patm']):
    """Sets the ambient air conditions via the dry bulb temperature and relative humidity (equal-length vectors)

    Parameters:
        TDryBulb        Dry bulb temperature of ambient air [C]
        RelHumidity     Relative humidity [0, 1]
        Pressure        (optional; defaults to atmospheric pressure, 101325 Pa) Pressure [Pa]

    Notes:
        CT operation is calculated in steady-state for all the ambient air conditions.

    Return:
        air_i           Ambient air, as tuple of humidair objects
        WBT             Ambient wet bulb temperature [C] as np array
    """
    if len(TDryBulb) != len(RelHumidity):
        raise ValueError("Parameters 'TDryBulb' and 'RelHumidity' must be of the same length.")

    air_i = tuple(humidair.fixstatefr_Tdb_RH_P(_T, _RH, P=Pressure) for _T, _RH in zip(TDryBulb, RelHumidity))
    WBT = np.fromiter((_air.TWetBulb for _air in air_i), dtype='f8')

    return air_i, WBT


def simulate_CT(heat_load, CT_design, air_i, exhaust_RH=0.95, fixedCWT_ctrl=False, pump_ctrl='flow limit',
                fan_ctrl=True, ignore_CT_eff=False):
    """Simulates CT operation. The water circulation and air heat transfer are solved sequentially*.

    2-dim tables (time x CT):
        Two dimensional tables have rows representing time (as ambient conditions change, represented by air_i),
        and columns representing all the CTs (ordered as in CT_design).

    Parameters:
        heat_load           Heat load of the CTs (time x CT) as Pint numpy array
        CT_design           DataFrame of CTs and their specs
        air_i               Tuple of ambient air conditions to solve
        exhaust_RH          (optional; defaults to 0.95 (95% relative humidity)) Exhaust air relative humidity
        ignore_CT_eff       (optional; defaults to False) If True, then the max CT efficiency limit is ignored. This
                            can be warranted to explore very hot and humid conditions (results might not be realistic).

    Control Parameters:
        fixedCWT_ctrl       (Water circulation) If True, CWT is set to design value. Else, CWT is calculated from
                            the specified CT performance curves.

        pump_ctrl           (Water circulation) Four modes:
                            'flow limit'    Water flow is limited to the rated limits of the CT ('Max(Min) per unit
                                            water flow' of CT design) This appears to be the most robust control (in
                                            that it does not result in extreme values of other parameters).

                            'range limit'   HWT is fixed to the design value, except when the resulting range is
                                            below the minimimum. In which case, HWT is increased until the minimum
                                            Range is respected. This is similar to 'fixed HWT' control in part-load
                                            conditions, but more robust to larger than designed WBT. Increasing HWT
                                            tends to increase the exhaust temperature as well.


                            'fixed HWT'     HWT is fixed to design value. This control allows for less water mass
                                            flow (and hence air) under part-load conditions, but would lead to higher
                                            flows when WBT increases. The control is compromised when significantly
                                            higher WBT is experienced than the design.

                            None            (Python None) Stands for the uncontrolled mode, where the HWT is allowed
                                            to float at the designed water flow rate.


        fan_ctrl            Water-air heat transfer -- if True, then the mass flow rate is adjusted such that the
                            exhaust air state is as designed. Else, the mass flow rate is as designed.

    Return:
        Dictionary of various output information, with the ff. key, value pairs:

            'air flow'      Air flow [kg/s] (time x CT)

    Notes:
        *Because the water circulation loop and water-air heat transfer are solved sequentially, assumptions made in
        the first step that prove to be invalid after the 2nd step are not remedied (soln fails).
    """
    nCT = CT_design.shape[0]
    nTime = len(air_i)
    if heat_load.shape != (nTime, nCT):
        raise ValueError('Inconsisent input dimensions. Pls. check air_i, heat_load and CT_design.')

    # ............................................................................ a) Get WBT and DBT (ambient)
    WBT = getstate('TWetBulb', air_i)
    DBT = getstate('TDryBulb', air_i)

    # ............................................................................ b) Calc water circulation
    # Note: No load condition is triggered here.
    res = _calc_WaterCirculation(heat_load, CT_design, WBT, DBT, fixedCWT_ctrl, pump_ctrl, ignore_CT_eff=ignore_CT_eff)
    # ............................................................................ c) Calc water-air heat transfer
    airflow, ret_waterflow, air_o, thermo = _calc_HeatTransfer(CT_design, air_i, res['HWT'], res['CWT'],
                                                               res['water flow'], fan_ctrl, exhaust_RH)

    # ............................................................................ d) Calc exhaust air speed
    volumetric_flow, speed = _calc_AirFlow(CT_design, air_o, airflow)

    Results = {
        'air_o': air_o,
        'air flow': airflow,
        'return water flow': ret_waterflow,
        'thermo': thermo,
        'exhaust volumetric flow': volumetric_flow,
        'exhaust speed': speed,
    }
    Results.update(res)
    return Results


def _calc_WaterCirculation(heat_load, CT_design, WBT, DBT, fixedCWT_ctrl, pump_ctrl, ignore_CT_eff, max_CT_eff=0.85):
    """Calculates the water circulation loop. Used by simulate_CT().

    Parameters:

    Returns:
        All (time x CT) arrays as

        HWT             Hot water temp [pint, C]
        CWT             Cold water temp [pint, C]
        waterflow       Water mass flow rate  [pint, kg/s]. This is the input water stream to the CTs.

    Notes:
        1) This routine determines the temperatures of the water circuit (HWT, CWT) and the water flow rate to
        transfer the heat load to the CT.

        2) The WBT serves as a lower limit to CWT.
        (variables: WBT is an iterable (length nTime), whereas WBT2 is a 2d array (time x CT))

    """
    nTime = len(WBT)
    nCT = CT_design.shape[0]
    # .......................................................... 1) Calc CWT (based on WBT) and approach
    # i) CWT
    if fixedCWT_ctrl:
        raise NotImplementedError
    # This ctrl is not as simple as setting CWT to rated, because what if ambient WBT + min approach is above this?
    # CWT fixed at design value
    # CWT = Q_(np.tile(CT_design['CWT [°C]'].values, (Nsimul, 1)), 'degC')
    else:
        # CWT from CT performance curves
        perf_m = CT_design['CT perf slope'].values
        perf_b = CT_design['CT perf y-int'].values

        # time x CT
        CWT = Q_(np.outer(WBT, perf_m) + np.tile(perf_b, (nTime, 1)), 'degC')

    # ii) Approach
    WBT2 = Q_(np.transpose(np.tile(WBT, (nCT, 1))), 'degC')
    approach = CWT - WBT2

    # .......................................................... 2) Calc water circulation loop
    #                                                              (calc deltaT, waterflow, assuming loaded)
    # Forms a time-invariant array with shape (time x CT) and as a Pint quantity
    tile_and_pint = lambda arr, units: Q_(np.tile(arr, (nTime, 1)), units)

    HWT_r = tile_and_pint(CT_design['HWT [°C]'].values, 'degC')
    waterflow_r = tile_and_pint(CT_design['water flow [kg/s]'].values, 'kg/s')

    if pump_ctrl == 'fixed HWT':
        deltaT = HWT_r - CWT
        waterflow = (heat_load / (cp_water * deltaT)).to_base_units()

    elif pump_ctrl == 'range limit':
        # Calc range as if HWT = HWT_r
        deltaT = HWT_r - CWT

        # i) Adjust deltaT
        deltaT_min = np.tile(CT_design['Min Range [C°]'].values, (nTime, 1))
        deltaT = Q_(np.clip((deltaT).magnitude, deltaT_min, None), 'delta_degC')

        # ii) Calc water flow
        waterflow = (heat_load / (cp_water * deltaT)).to_base_units()

    elif pump_ctrl == 'c':
        # Calc range & water flow as if HWT = HWT_r
        deltaT = HWT_r - CWT
        waterflow = (heat_load / (cp_water * deltaT)).to_base_units()
        waterflow_units = waterflow.units

        # i) Adjust water flow
        # Clip violating values
        waterflow_ub = np.tile((CT_design['Max per unit water flow'] * CT_design['water flow [kg/s]']).values,
                               (nTime, 1))
        waterflow_lb = np.tile((CT_design['Min per unit water flow'] * CT_design['water flow [kg/s]']).values,
                               (nTime, 1))

        _wf = np.clip(waterflow.magnitude, waterflow_lb, waterflow_ub)
        # Back to pint
        waterflow = Q_(_wf, waterflow_units)

        # ii) Calc deltaT
        deltaT = (heat_load / (cp_water * waterflow)).to('delta_degC')

    else:
        waterflow = waterflow_r
        deltaT = (heat_load / (cp_water * waterflow)).to('delta_degC')

    # .......................................................... 3) No-load fix
    # This part is necessary for all conrtol modes because the operational limits applied
    # in the step 2 assumed loaded operation. After this step, water flow and deltaT are final.
    CT_load_mask = (heat_load != 0).astype('int')  # 0 if no load, 1 otherwise
    waterflow = waterflow * CT_load_mask
    deltaT = deltaT * CT_load_mask
    HWT = CWT + deltaT

    # .......................................................... 4) HWT and CWT adjustment
    # HWT cannot be less than DBT; in which case, HWT is limited to DBT and CWT rises.
    # Vectorize DBT into (time x CT)
    DBT = np.tile(DBT, (nCT, 1)).transpose()

    HWT = Q_(np.maximum(HWT.magnitude, DBT), 'degC')
    CWT = HWT - deltaT

    # .......................................................... 5) Checks and return
    assert waterflow.units == ureg.kg / ureg.s
    assert deltaT.units == ureg.delta_degC, deltaT.units

    # Check that CT efficiency is realistic. In practice, efficiency is 65-70% (normal operating conditions)
    CT_eff = deltaT / (deltaT + approach)
    assert ignore_CT_eff or np.all(CT_eff < max_CT_eff), \
        "CT efficiency exceeded the limit: {}".format(CT_eff)

    assert all(obj.shape == (nTime, nCT) for obj in (HWT, CWT, waterflow, deltaT, approach, CT_eff))
    # Check energy balance
    assert np.allclose(heat_load.magnitude, (cp_water * deltaT * waterflow).to(heat_load.units).magnitude)

    res = {
        'HWT': HWT,
        'CWT': CWT,
        'water flow': waterflow,
        'range': deltaT,
        'approach': approach,
        'CT_eff': CT_eff,
    }

    return res


def _calc_HeatTransfer(CT_design, air_i, HWT, CWT, waterflow, fan_ctrl, exhaust_RH):
    """Calculates the heat transfer from the water to the air streams.

    Return:
        airflow             Air mass flow rate (time x CT) (pint kg/s)
        ret_waterflow       Output water stream flow rate (time x CT) (pint kg/s)

        air_o               DataFrame (time x CT with range index) of the exhaust humid air objects
                            Note: At no-load conditions, this maps to the appropriate ambient air conditions.

        thermo              Dictionary of the thermodynamic states (all time x CT as pint 2d arrays):
                                h1, h2          Enthalpy of inlet and outlet water streams, respectively.
                                h3, w3          Enthalpy and humidity ratio of ambient air
                                h4, w4          Enthalpy and humidity ratio of exhaust air
    """
    nCT = CT_design.shape[0]
    nTime = len(air_i)
    _shape = (nTime, nCT)  # time x CT
    # Thermodynamic properties (h, w) are all 2d vec (time x CT)
    # ----------------------------------------------------------------- a) Enthalpies of water
    # h1 (HWT) - varies with Twb(time) if w/o pump control
    # h2 (CWT) - varies with Twb(time) and with CT
    h1 = Q_(liqwater_h(HWT.magnitude), 'J/kg')
    h2 = Q_(liqwater_h(CWT.magnitude), 'J/kg')

    # ----------------------------------------------------------------- b) Enthalpy and humidity ratio of ambient air
    # h3 (amb) - varies with Twb(time)
    col = np.fromiter((_air.h_moist for _air in air_i), dtype='f8')
    h3 = Q_(np.transpose(np.tile(col, (nCT, 1))), 'J/kg')

    # w3
    col = np.fromiter((_air.HumRatio for _air in air_i), dtype='f8')
    w3 = Q_(np.transpose(np.tile(col, (nCT, 1))), ' ')

    # ----------------------------------------------------------------- c) Exhaust air stream (fan control)
    #    if w/ control, state 4 is determined; solve air flow
    #    else, air flow as rated; solve state 4
    ignore_div_0 = False

    if fan_ctrl:
        # ...................................................... c-i) w/ control
        h4 = np.empty(_shape)
        w4 = np.empty(_shape)

        # Compact storage of exhaust air states, to prevent redundant state-fixing
        air_o_cmpt = {}
        # DataFrame (time x CT range index)
        air_o = pd.DataFrame(index=pd.RangeIndex(_shape[0]), columns=pd.RangeIndex(_shape[1]))

        for t_idx in range(_shape[0]):
            for CT_idx in range(_shape[1]):
                # i) Get state 4
                # a) No-load --> set to ambient
                if h1[t_idx, CT_idx] == h2[t_idx, CT_idx]:
                    _air = air_i[t_idx]
                    ignore_div_0 = True

                # b) Loaded --> set T4 = HWT at assumed RH
                else:
                    TRH_key = (HWT[t_idx, CT_idx].magnitude, exhaust_RH)

                    if TRH_key not in air_o_cmpt:
                        air_o_cmpt[TRH_key] = humidair.fixstatefr_Tdb_RH_P(*TRH_key, P=c_['Patm'])

                    _air = air_o_cmpt[TRH_key]

                # ii) Get enthalpy and humidity ratio
                h4[t_idx, CT_idx] = _air.h_moist
                w4[t_idx, CT_idx] = _air.HumRatio

                # iii) map to air_o (DataFrame)
                air_o.at[t_idx, CT_idx] = _air

        # Convert to Pint
        h4 = Q_(h4, 'J/kg')
        w4 = Q_(w4, ' ')

    else:
        # ...................................................... c-ii) w/o control
        # i) Set air flow (rated)
        airflow = Q_(np.tile(CT_design['air flow [kg/s]'].values, (nTime, 1)), 'kg/s')

    # ----------------------------------------------------------------- d) Energy and mass balance
    # Energy balance to calculate air stream
    if fan_ctrl:
        if ignore_div_0: np.seterr(divide='ignore', invalid='ignore')
        # At no load, this corresponds to 0/0 (state 1 == state 2 and state 3 == state 4)
        airflow = waterflow * (h1 - h2) / ((h4 - h3) + (w3 - w4) * h2)
        airflow_units = airflow.units

        airflow = Q_(np.nan_to_num(airflow.magnitude), airflow_units)
        np.seterr(divide='warn', invalid='warn')
    else:
        # Solve for T4 from the energy balance numerically (T4 --> h4, w4)
        raise NotImplementedError

    # Mass balance to calculate returning water flow
    ret_waterflow = waterflow - (w4 - w3) * airflow

    # ----------------------------------------------------------------- e) Checks and return
    assert all(obj.shape == _shape for obj in (h1, h2, h3, h4, w3, w4, airflow, ret_waterflow))
    assert all(obj.units == ureg.kg / ureg.s for obj in (airflow, ret_waterflow))
    assert all(obj.units == ureg.J / ureg.kg for obj in (h1, h2, h3, h4))
    assert np.allclose((waterflow * h1 + airflow * h3).magnitude, (ret_waterflow * h2 + airflow * h4).magnitude)
    assert np.all(airflow >= 0)

    thermo = {
        'h1': h1,
        'h2': h2,
        'h3': h3,
        'h4': h4,
        'w3': w3,
        'w4': w4,
    }

    return airflow, ret_waterflow, air_o, thermo


def _calc_AirFlow(CT_design, air_o, airflow):
    """Calculates the exhaust volumetric flow and speed.

    speed = mass flow * sp. volume / fan area
            [kg/s]      [m^3/kg]     [m^2]
    """
    nTime = airflow.shape[0]

    # 1) Specific volume
    sp_vol = get_exhaust_air_state('MoistAirVolume', air_o, units='m^3/kg')

    # 2) Fan diameter
    fan_dia = Q_(np.tile(CT_design['Fan diameter [m]'].values, (nTime, 1)), 'm')
    fan_area = np.pi * (fan_dia / 2) ** 2

    # 3) Calc
    volumetric_flow = airflow * sp_vol
    speed = volumetric_flow / fan_area

    return volumetric_flow, speed


def get_exhaust_air_state(state, air_o, units=None):
    """Returns a 2d-array (time x CT) of the specified state of air_o"""
    nTime, nCT = air_o.shape

    # This works too but is less elegant / more difficult to read
    # air_o_states = np.fromiter((getattr(air_o.iat[i, j], state) for i in range(nTime) for j in range(nCT)),
    #                            dtype='f8').reshape((nTime, nCT))
    air_o_states = np.array([[getattr(air_o.iat[i, j], state) for j in range(nCT)] for i in range(nTime)])

    if units:
        air_o_states = Q_(air_o_states, units)

    return air_o_states


# ----------------------------------------------------- PLOTTING ----------------------------------------------------- #
def plt_AmbientAirPerformance_exhaust(state, results, Tin, RH_values, pu_load, pump_ctrl,
                                      plot_setpoint=False, save_as=None, **kwargs):
    """Plots the ambient air performance -- exhaust air state vs. T ambient

    This plots the target variable vs T ambient parametrized by RH.

    Parameters:
        state           State of humidair (currently support: TDryBulb and HumRatio)

        results         Dict of {pump_ctrl, RH, *: val}, where * are the standard keys in simulate_CT(); and val
                        are the results objects returned.

        Tin             The sequence of ambient air temperatures [°C] (independent variable)

        RH_values       The sequence of relative humidity values [0,1]

        pu_load         CT load in per unit [0,1]

        pump_ctrl       Boolean flag for pump control (part of results key)

        plot_setpoint   (Optional; defaults to False). Plots the set-point exhaust state according to the designed
                        value. This is a constant T- and w-line. If True, then need to provide the ff. kwargs:
                        'T_sp' and 'w_sp' as the temperature and relative humidity set points,
                        respectively.


        kwargs          Plot kwargs
    """
    def_ylabels = {
        'TDryBulb': 'Temp (dry bulb) [°C]',
        'HumRatio': '[kg vapor/kg d.a.]',
    }
    def_titles = {
        'TDryBulb': 'Dry Bulb Temperature',
        'HumRatio': 'Humidity Ratio',
        1: 'Full',
        0: 'No',
    }
    def_kwargs = {
        'title': 'Exhaust {} at {} Load'.format(def_titles.get(state, '*'),
                                                def_titles.get(pu_load, '{:0.1f}%'.format(pu_load * 100))),
        'ylabel': '{}'.format(def_ylabels.get(state, '*')),
        'xlabel': 'Ambient Temp (dry bulb) [°C]',
        'setpoint_line': {'ls': '--', 'lw': 1, 'color': 'k'},
    }

    kwargs.update({key: val for key, val in def_kwargs.items() if key not in kwargs})
    kwargs.update({key: val for key, val in common_def_kwargs.items() if key not in kwargs})

    RH_color_seq = ('#2E86C1', '#16A085', '#D35400')

    # ----------------------------------------------------- PLOT
    plt.figure(figsize=kwargs['figsize'])
    for idx, RH in enumerate(RH_values):
        plt.plot(Tin, getstate(state, results[pump_ctrl, RH, 'air_o']),
                 label='{:0.2f} RH'.format(RH), color=RH_color_seq[idx])

    ax = plt.gca()
    ax = basic_plot_polishing(ax, **kwargs)

    if plot_setpoint:
        setpoint = kwargs[{'TDryBulb': 'T_sp', 'HumRatio': 'w_sp'}[state]]
        ax.axhline(setpoint, **kwargs['setpoint_line'])

        # Text label
        y_lb, y_ub = ax.get_ylim()
        text_y = setpoint + 0.03 * (y_ub - y_lb)
        if text_y > y_ub * 0.95: text_y = setpoint - 0.03 * (y_ub - y_lb)

        plt.text(Tin.min(), text_y, 'set point')

    if save_as:
        plt.savefig(path.join(PathPlots, save_as), dpi=kwargs.get('dpi'))

    plt.show()
    return


def plt_AmbientAirPerformance_airflow(results, Tin, RH_values, pu_load, pump_ctrl, plot_setpoint=True,
                                      save_as=None, **kwargs):
    """Plots the ambient air performance -- air flow vs. T ambient"""
    def_kwargs = {
        'title': 'Air Mass Flow at {} Load'.format({1: 'Full', 0: 'No'}.get(pu_load, '{:0.1f}%'.format(pu_load * 100))),
        'ylabel': '[kg/s]',
        'xlabel': 'Temp (dry bulb) [°C]',
        'setpoint_line': {'ls': '--', 'lw': 1, 'color': 'k'},
    }
    kwargs.update({key: val for key, val in def_kwargs.items() if key not in kwargs})
    kwargs.update({key: val for key, val in common_def_kwargs.items() if key not in kwargs})

    RH_color_seq = ('#2E86C1', '#16A085', '#D35400')

    # ----------------------------------------------------- PLOT
    plt.figure(figsize=kwargs['figsize'])

    for idx, RH in enumerate(RH_values):
        plt.plot(Tin, results[pump_ctrl, RH, 'air flow'].magnitude,
                 label='{:0.2f} RH'.format(RH), color=RH_color_seq[idx])

    ax = plt.gca()
    ax = basic_plot_polishing(ax, **kwargs)

    if plot_setpoint:
        setpoint = kwargs['airflow_sp']
        ax.axhline(setpoint, **kwargs['setpoint_line'])

        # Text label
        y_lb, y_ub = ax.get_ylim()
        text_y = setpoint + 0.03 * (y_ub - y_lb)
        if text_y > y_ub * 0.95: text_y = setpoint - 0.03 * (y_ub - y_lb)

        plt.text(Tin.min(), text_y, 'nominal')

    if save_as:
        plt.savefig(path.join(PathPlots, save_as), dpi=kwargs.get('dpi'))

    plt.show()
    return


def plt_LoadingPerformance_exhaust(state, results, CT_load, air_i, pump_ctrl, plot_setpoint=True, save_as=None,
                                   **kwargs):
    """Plots the loading performance -- exhaust vs. load kW"""
    def_ylabels = {
        'TDryBulb': 'Temp (dry bulb) [°C]',
        'HumRatio': '[kg vapor/kg d.a.]',
    }
    def_titles = {
        'TDryBulb': 'Temperature',
        'HumRatio': 'Humidity Ratio',
    }
    def_kwargs = {
        'xlabel': 'heat load [kW]',
        'ylabel': '{}'.format(def_ylabels.get(state, '*')),
        'title': 'Exhaust {} vs. Load'.format(def_titles.get(state, '*')),
        'setpoint_line': {'ls': '--', 'lw': 1, 'color': 'k'},
    }
    kwargs.update({key: val for key, val in def_kwargs.items() if key not in kwargs})
    kwargs.update({key: val for key, val in common_def_kwargs.items() if key not in kwargs})

    RH_color_seq = ('#2E86C1', '#16A085', '#D35400')

    # ----------------------------------------------------- PLOT
    plt.figure(figsize=kwargs['figsize'])

    for idx, _air_i in enumerate(air_i):
        _T, _RH = _air_i.TDryBulb, _air_i.RelHum
        plt.plot(CT_load.magnitude, getstate(state, results[_T, _RH, pump_ctrl, 'air_o']),
                 label='{:0.1f}°C, {:0.3f} RH'.format(_T, _RH), color=RH_color_seq[idx])

    ax = plt.gca()

    ax = basic_plot_polishing(ax, **kwargs)

    if plot_setpoint:
        setpoint = kwargs[{'TDryBulb': 'T_sp', 'HumRatio': 'w_sp'}[state]]
        ax.axhline(setpoint, **kwargs['setpoint_line'])

        # Text label
        y_lb, y_ub = ax.get_ylim()
        text_y = setpoint + 0.03 * (y_ub - y_lb)
        if text_y > y_ub * 0.95: text_y = setpoint - 0.03 * (y_ub - y_lb)

        plt.text(0, text_y, 'set point')

    if save_as:
        plt.savefig(path.join(PathPlots, save_as), dpi=kwargs.get('dpi'))

    plt.show()
    return


def plt_LoadingPerformance_airflow(results, CT_load, air_i, pump_ctrl, plot_setpoint=True, save_as=None, **kwargs):
    """Plots the loading performance -- air flow vs. load kW"""
    def_kwargs = {
        'xlabel': 'heat load [kW]',
        'ylabel': '[kg/s]',
        'title': 'Air Mass Flow vs. Load',
        'setpoint_line': {'ls': '--', 'lw': 1, 'color': 'k'},
    }
    kwargs.update({key: val for key, val in def_kwargs.items() if key not in kwargs})
    kwargs.update({key: val for key, val in common_def_kwargs.items() if key not in kwargs})

    RH_color_seq = ('#2E86C1', '#16A085', '#D35400')

    # ----------------------------------------------------- PLOT
    plt.figure(figsize=kwargs['figsize'])

    for idx, _air_i in enumerate(air_i):
        _T, _RH = _air_i.TDryBulb, _air_i.RelHum
        plt.plot(CT_load.magnitude, results[_T, _RH, pump_ctrl, 'air flow'].magnitude,
                 label='{:0.1f}°C, {:0.3f} RH'.format(_T, _RH), color=RH_color_seq[idx])

    ax = plt.gca()
    ax = basic_plot_polishing(ax, **kwargs)

    if plot_setpoint:
        setpoint = kwargs['airflow_sp']
        ax.axhline(setpoint, **kwargs['setpoint_line'])

        # Text label
        y_lb, y_ub = ax.get_ylim()
        text_y = setpoint + 0.03 * (y_ub - y_lb)
        if text_y > y_ub * 0.95: text_y = setpoint - 0.03 * (y_ub - y_lb)

        plt.text(0, text_y, 'nominal')

    if save_as:
        plt.savefig(path.join(PathPlots, save_as), dpi=kwargs.get('dpi'))

    plt.show()
    return


def plt_ExhaustSpeeds(results, CT_selection, load_levels_pu, amb_T_RH, pump_ctrl, save_as=None, **kwargs):
    """Plots the exhaust speeds of multiple CTs (intended for the largest CT per fan size)"""
    def_kwargs = {
        'xlabel': 'Load [%]',
        'ylabel': '[m/s]',
        'title': 'Exhaust Air Speed vs. Load',
        'legend_kw': {'loc': 'lower right', 'title': 'CT size and fan diameter'},
    }
    kwargs.update({key: val for key, val in def_kwargs.items() if key not in kwargs})
    kwargs.update({key: val for key, val in common_def_kwargs.items() if key not in kwargs})

    nCT = CT_selection.shape[0]
    CT_color_seq = ('#5499C7', '#52BE80', '#F39C12', '#E74C3C', '#8E44AD', '#839192', '#2E4053')
    Tamb, RHamb = amb_T_RH

    # ----------------------------------------------------- PLOT
    plt.figure(figsize=kwargs['figsize'])

    for CTidx in range(nCT):
        plt.plot(load_levels_pu * 100, results[Tamb, RHamb, pump_ctrl, 'exhaust speed'][:, CTidx].magnitude,
                 label='{} kW, {} m'.format(CT_selection['Capacity [kW]'].iat[CTidx],
                                            CT_selection['Fan diameter [m]'].iat[CTidx]),
                 color=CT_color_seq[CTidx], )

    ax = plt.gca()
    ax = basic_plot_polishing(ax, **kwargs)
    plt.text(0.86, 0.42, 'Ambient Conditions', fontdict={'fontweight': 0}, horizontalalignment='center',
             transform=ax.transAxes)
    plt.text(0.86, 0.37, '{}°C, {} RH'.format(Tamb, RHamb), horizontalalignment='center', transform=ax.transAxes)

    if save_as:
        plt.savefig(path.join(PathPlots, save_as), dpi=kwargs.get('dpi'))

    plt.show()
    return


# --------------------------------------------------- AUX SCRIPTS --------------------------------------------------- #
def parse_BldgToCTs(BldgToCTs, CT_catalog):
    """Creates the CT dataframe to for simulate_CT(), which contains units from CT_catalog under the mapping defined
    by BldgToCTs. The CT order is defined here, and is based on on the order of BldgToCTs.

    CT ORDER:
        Follows this nested loop structure: 1) for bldg in CT_catalog; 2) for CT in bldg

    PARAMETERS:
        BldgToCTs       Dictionary {bldg id: tuple of CT kW capacities that comprise the total CT system}
                        e.g. {0: (5000, 5000, 5000)} for bldg 0 which has 3x 5000-kW CTs

        CT_catalog      DataFrame; catalog of CT models (only one model per kW rating is allowed; indexed by the kW
                        rating). The CT catalog must contain all the fields required by simulate_CT().

    RETURNS:
        CTs DataFrame, which has the same structure as CT_catalog, whose rows correspond to the CTs of the bldgs (
        ordered as they appear per bldg, as well as for all bldgs).
    """
    AllCTs = pd.DataFrame(columns=CT_catalog.columns, dtype='f8')  # the index actually doesn't matter

    for CTs_of_bldgT in BldgToCTs.values():
        for CT_kW in CTs_of_bldgT:
            AllCTs.loc[AllCTs.shape[0], :] = CT_catalog.loc[CT_kW, :]

    return AllCTs


def find_upper_neighbours(value, df, colname):
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        upperneighbour = df[colname][upperneighbour_ind]
        return upperneighbour

def size_cooling_tower(group_demand_df, CT_catalog, BASE_CT_THRESHOLD, OVERDIMENSIONING_THRESHOLD):
    BldgToCTs = {}
    for (group, demand) in group_demand_df.iteritems():
        peak = max(demand)

        #every CT has three main units
        #1. a base unit
        baseload = BASE_CT_THRESHOLD * peak
        base_unit_size = baseload * (1+OVERDIMENSIONING_THRESHOLD)
        #2. a intermediate unit
        average = np.mean(demand.replace(0, np.NaN))
        intermediate_unit_size = (average - baseload) * (1+OVERDIMENSIONING_THRESHOLD)
        #3. a peak unit
        peak_unit_size = (peak - average) * (1+OVERDIMENSIONING_THRESHOLD)


        # checks
        if peak_unit_size > CT_catalog['Capacity [kW]'].min():
            peak_unit_size = find_upper_neighbours(peak_unit_size, CT_catalog, 'Capacity [kW]')
        else:
            peak_unit_size = CT_catalog['Capacity [kW]'].min()

        if intermediate_unit_size > CT_catalog['Capacity [kW]'].min():
            intermediate_unit_size = find_upper_neighbours(intermediate_unit_size, CT_catalog, 'Capacity [kW]')
        else:
            intermediate_unit_size = CT_catalog['Capacity [kW]'].min()

        if base_unit_size > CT_catalog['Capacity [kW]'].min():
            base_unit_size = find_upper_neighbours(base_unit_size, CT_catalog, 'Capacity [kW]')
        else:
            base_unit_size = CT_catalog['Capacity [kW]'].min()

        BldgToCTs[group] = peak_unit_size, intermediate_unit_size, base_unit_size
    return BldgToCTs

def calc_CTheatload(BldgDemand, BldgToCTs, CT_catalog, t_from=None, t_to=None):
    """Calculates the CT heat load to be passed to simulate_CT() by allocating the total heat load to all the CTs by
    their capacity (i.e. no smart scheduling of CTs accdg. to load).

    PARAMETERS:
        BldgDemand          DataFrame of the heat rejected by the bldgs [kW]. The index is interpretted as time,
                            and the column names are interpretted as the bldg indices.

        BldgToCTs           Dictionary of {bldg id : tuple(CT ids)}. This is a mapping of which CT units in
                            CT_catalog are used by the buldings.

        CT_catalog          DataFrame; catalog of CT models

        t_from, t_to        Time slice of BldgDemand to calculate the heat load (inclusive).

    RETURNS:
        Heat load (pint; kW)

    """
    # ----------------------------------------------------------------------------- 0) Init
    if tuple(BldgToCTs.keys()) != tuple(BldgDemand.columns):
        raise ValueError('Inconsistent building indices. The keys should match, including their order.')

    if t_from is None:
        t_from = BldgDemand.index[0]
    if t_to is None:
        t_to = BldgDemand.index[-1]

    # HL 2D init
    nTime = len(BldgDemand.loc[t_from:t_to])
    nCT = sum(len(bldg_CTs) for bldg_CTs in BldgToCTs.values())
    CT_heat_load = np.zeros((nTime, nCT), dtype='f8')

    # ----------------------------------------------------------------------------- 1) Populate the heat load
    CT_HL_col = 0
    groups = []
    for bldg_idx, bldg_CTs in BldgToCTs.items():
        # Slice to get bldg heat load
        Total_HL = BldgDemand.loc[t_from:t_to, bldg_idx].to_numpy()

        # Get CT kW sizes
        CT_sizes = np.fromiter((CT_catalog.at[CTidx, 'Capacity [kW]'] for CTidx in bldg_CTs), dtype='f8')
        CT_wts = CT_sizes / CT_sizes.sum()

        # Write to 2D array
        for wt in CT_wts:
            CT_heat_load[:, CT_HL_col] = wt * Total_HL
            CT_HL_col += 1

        # groups and cooling towers order
        groups.extend([bldg_idx] * 3)

    # ----------------------------------------------------------------------------- 2) Check and return as pint
    assert CT_HL_col == nCT
    assert abs(CT_heat_load.sum() - BldgDemand.loc[t_from:t_to].sum().sum()) < 10 ** -6

    return Q_(CT_heat_load, 'kW'), groups

common_def_kwargs = {
    'figsize': (9.6, 6),
    'legend': True,
    'xlabel_kw': {'fontsize': 12.5, },
    'xticks_kw': {'fontsize': 11},
    'ylabel_kw': {'fontsize': 12.5, },
    'yticks_kw': {'fontsize': 11},
    'title_kw': {'fontsize': 13},
    'dpi': 300,
}
