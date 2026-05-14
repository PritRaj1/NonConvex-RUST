use crate::utils::opt_prob::FloatNumber;

pub trait CoolingSchedule<T: FloatNumber> {
    fn temperature(&self, initial_temp: T, iteration: usize, cooling_rate: T) -> T;
    fn reheat(&self, initial_temp: T) -> T;
    fn adaptive_temperature(
        &self,
        initial_temp: T,
        iteration: usize,
        cooling_rate: T,
        success_rate: f64,
    ) -> T;
}

pub struct ExponentialCooling;

impl<T: FloatNumber> CoolingSchedule<T> for ExponentialCooling {
    fn temperature(&self, initial_temp: T, iteration: usize, cooling_rate: T) -> T {
        initial_temp * cooling_rate.powi(iteration as i32)
    }

    fn reheat(&self, initial_temp: T) -> T {
        initial_temp * T::cast(0.8)
    }

    fn adaptive_temperature(
        &self,
        initial_temp: T,
        iteration: usize,
        cooling_rate: T,
        success_rate: f64,
    ) -> T {
        let base_temp = initial_temp * cooling_rate.powi(iteration as i32);
        if success_rate < 0.2 {
            base_temp * T::cast(1.2) // Increase when low success, explore more
        } else if success_rate > 0.6 {
            base_temp * T::cast(0.9) // Decrease when high success, exploit more
        } else {
            base_temp
        }
    }
}

pub struct LogarithmicCooling;

impl<T: FloatNumber> CoolingSchedule<T> for LogarithmicCooling {
    fn temperature(&self, initial_temp: T, iteration: usize, _cooling_rate: T) -> T {
        initial_temp / T::cast(1.0 + (iteration as f64).ln())
    }

    fn reheat(&self, initial_temp: T) -> T {
        initial_temp * T::cast(0.7)
    }

    fn adaptive_temperature(
        &self,
        initial_temp: T,
        iteration: usize,
        _cooling_rate: T,
        success_rate: f64,
    ) -> T {
        let base_temp = initial_temp / T::cast(1.0 + (iteration as f64).ln());
        if success_rate < 0.2 {
            base_temp * T::cast(1.3)
        } else if success_rate > 0.6 {
            base_temp * T::cast(0.85)
        } else {
            base_temp
        }
    }
}

pub struct CauchyCooling;

impl<T: FloatNumber> CoolingSchedule<T> for CauchyCooling {
    fn temperature(&self, initial_temp: T, iteration: usize, _cooling_rate: T) -> T {
        initial_temp / T::cast(1.0 + iteration as f64)
    }

    fn reheat(&self, initial_temp: T) -> T {
        initial_temp * T::cast(0.75)
    }

    fn adaptive_temperature(
        &self,
        initial_temp: T,
        iteration: usize,
        _cooling_rate: T,
        success_rate: f64,
    ) -> T {
        let base_temp = initial_temp / T::cast(1.0 + iteration as f64);
        if success_rate < 0.2 {
            base_temp * T::cast(1.4)
        } else if success_rate > 0.6 {
            base_temp * T::cast(0.8)
        } else {
            base_temp
        }
    }
}

pub struct AdaptiveCooling;

// Start with exponential, then adapt based on iter
impl<T: FloatNumber> CoolingSchedule<T> for AdaptiveCooling {
    fn temperature(&self, initial_temp: T, iteration: usize, cooling_rate: T) -> T {
        if iteration < 100 {
            initial_temp * cooling_rate.powi(iteration as i32)
        } else {
            initial_temp * cooling_rate.powi((iteration / 2) as i32)
        }
    }

    fn reheat(&self, initial_temp: T) -> T {
        initial_temp * T::cast(0.6)
    }

    fn adaptive_temperature(
        &self,
        initial_temp: T,
        iteration: usize,
        cooling_rate: T,
        success_rate: f64,
    ) -> T {
        let base_temp = self.temperature(initial_temp, iteration, cooling_rate);

        let adaptation_factor = if success_rate < 0.1 {
            T::cast(2.0) // Significant increase for very low success, explore more
        } else if success_rate < 0.3 {
            T::cast(1.5) // Moderate increase for low success, explore more
        } else if success_rate > 0.7 {
            T::cast(0.7) // Decrease for high success, exploit more
        } else {
            T::one()
        };

        base_temp * adaptation_factor
    }
}
