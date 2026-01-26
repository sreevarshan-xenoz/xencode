"""
Lazy loading utilities for Xencode
Provides lazy loading for heavy components to optimize resource usage
"""
import importlib
import sys
from typing import Any, Callable, Optional, TypeVar
from functools import wraps
from threading import Lock


T = TypeVar('T')


class LazyLoader:
    """A lazy loader that defers the loading of heavy components until they are actually needed"""
    
    def __init__(self, module_name: str, attribute_name: str = None, callable_factory: Callable = None):
        """
        Initialize the lazy loader.
        
        Args:
            module_name: Name of the module to load
            attribute_name: Name of the attribute/class/function to load from the module
            callable_factory: Alternative factory function to create the object
        """
        self.module_name = module_name
        self.attribute_name = attribute_name
        self.callable_factory = callable_factory
        self._obj = None
        self._loaded = False
        self._lock = Lock()
    
    def _load(self) -> Any:
        """Actually load the component"""
        if self._loaded:
            return self._obj
        
        with self._lock:
            if self._loaded:  # Double-check locking pattern
                return self._obj
            
            if self.callable_factory:
                # Use the factory function to create the object
                self._obj = self.callable_factory()
            else:
                # Import the module and get the attribute
                module = importlib.import_module(self.module_name)
                if self.attribute_name:
                    self._obj = getattr(module, self.attribute_name)
                else:
                    self._obj = module
            
            self._loaded = True
            return self._obj
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the loaded object"""
        obj = self._load()
        return getattr(obj, name)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Make the lazy loader callable if the underlying object is callable"""
        obj = self._load()
        return obj(*args, **kwargs)
    
    @property
    def is_loaded(self) -> bool:
        """Check if the component has been loaded"""
        return self._loaded
    
    def force_load(self) -> Any:
        """Force loading of the component and return it"""
        return self._load()


def lazy_import(module_name: str, attribute_name: str = None):
    """
    Decorator to create a lazy-loaded import.
    
    Args:
        module_name: Name of the module to import lazily
        attribute_name: Name of the attribute to import from the module
    """
    def decorator(func_or_class):
        loader = LazyLoader(module_name, attribute_name)
        
        @wraps(func_or_class)
        def wrapper(*args, **kwargs):
            # Replace the decorated function/class with the lazy loader
            return loader(*args, **kwargs)
        
        return wrapper
    return decorator


class ComponentRegistry:
    """Registry for managing lazy-loaded components"""
    
    def __init__(self):
        self._components = {}
        self._locks = {}
    
    def register_lazy(self, name: str, module_name: str, attribute_name: str = None, callable_factory: Callable = None):
        """
        Register a component for lazy loading.
        
        Args:
            name: Name to register the component under
            module_name: Name of the module to load
            attribute_name: Name of the attribute to load from the module
            callable_factory: Alternative factory function to create the object
        """
        self._components[name] = LazyLoader(module_name, attribute_name, callable_factory)
        self._locks[name] = Lock()
    
    def get(self, name: str) -> Any:
        """
        Get a registered component (will be loaded if not already loaded).
        
        Args:
            name: Name of the component to get
            
        Returns:
            The loaded component
        """
        if name not in self._components:
            raise KeyError(f"Component '{name}' is not registered")
        
        return self._components[name]._load()
    
    def is_loaded(self, name: str) -> bool:
        """
        Check if a component is loaded.
        
        Args:
            name: Name of the component to check
            
        Returns:
            True if loaded, False otherwise
        """
        if name not in self._components:
            return False
        return self._components[name].is_loaded
    
    def force_load_all(self):
        """Force load all registered components"""
        for name in self._components:
            self.get(name)
    
    def get_component_info(self) -> dict:
        """Get information about all registered components"""
        info = {}
        for name, loader in self._components.items():
            info[name] = {
                'module_name': loader.module_name,
                'attribute_name': loader.attribute_name,
                'is_loaded': loader.is_loaded,
                'callable_factory': loader.callable_factory is not None
            }
        return info


# Global component registry
_component_registry = ComponentRegistry()


def register_component(name: str, module_name: str, attribute_name: str = None, callable_factory: Callable = None):
    """
    Register a component for lazy loading in the global registry.
    
    Args:
        name: Name to register the component under
        module_name: Name of the module to load
        attribute_name: Name of the attribute to load from the module
        callable_factory: Alternative factory function to create the object
    """
    _component_registry.register_lazy(name, module_name, attribute_name, callable_factory)


def get_component(name: str) -> Any:
    """
    Get a registered component from the global registry.
    
    Args:
        name: Name of the component to get
        
    Returns:
        The loaded component
    """
    return _component_registry.get(name)


def is_component_loaded(name: str) -> bool:
    """
    Check if a component is loaded in the global registry.
    
    Args:
        name: Name of the component to check
        
    Returns:
        True if loaded, False otherwise
    """
    return _component_registry.is_loaded(name)


# Pre-register some common heavy components that might be used in Xencode
register_component(
    'ollama_client',
    'requests',
    callable_factory=lambda: importlib.import_module('requests')
)

register_component(
    'rich_console',
    'rich.console',
    'Console',
    callable_factory=lambda: importlib.import_module('rich.console').Console()
)

# Example of how to register a more complex component
# register_component(
#     'model_manager',
#     'xencode.core.models',
#     'ModelManager',
#     callable_factory=lambda: importlib.import_module('xencode.core.models').ModelManager()
# )


class LazyProperty:
    """A descriptor to create lazy properties"""
    
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        self.lock = Lock()
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        if not hasattr(instance, f'_lazy_{self.name}'):
            with self.lock:
                # Double-check locking pattern
                if not hasattr(instance, f'_lazy_{self.name}'):
                    setattr(instance, f'_lazy_{self.name}', self.func(instance))
        
        return getattr(instance, f'_lazy_{self.name}')


def create_lazy_property(func: Callable) -> LazyProperty:
    """Decorator to create a lazy property"""
    return LazyProperty(func)


# Example usage classes
class HeavyComponentManager:
    """Example manager that uses lazy loading for heavy components"""
    
    def __init__(self):
        # Register some heavy components for lazy loading
        register_component(
            'heavy_nlp_model',
            'spacy',
            callable_factory=lambda: importlib.import_module('spacy').load('en_core_web_sm') if 'spacy' in sys.modules else None
        )
        
        register_component(
            'advanced_math_lib',
            'numpy',
            callable_factory=lambda: importlib.import_module('numpy')
        )
    
    @create_lazy_property
    def nlp_model(self):
        """Lazy property for NLP model"""
        print("Loading NLP model...")
        # In a real implementation, this would load a heavy NLP model
        return get_component('heavy_nlp_model')
    
    @create_lazy_property  
    def math_lib(self):
        """Lazy property for math library"""
        print("Loading math library...")
        return get_component('advanced_math_lib')
    
    def use_nlp_feature(self):
        """Method that uses the NLP model (loads it when first called)"""
        model = self.nlp_model  # This will load the model if not already loaded
        if model:
            print("Using NLP model...")
            # Use the model for processing
        else:
            print("NLP model not available")
    
    def use_math_feature(self):
        """Method that uses the math library (loads it when first called)"""
        lib = self.math_lib  # This will load the library if not already loaded
        if lib:
            print("Using math library...")
            # Use the library for computations
        else:
            print("Math library not available")


# Singleton instance for easy access
lazy_component_manager = HeavyComponentManager()