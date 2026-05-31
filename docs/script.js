/* 🌟 Script Premium - Blog Técnico: NeuralTransit Lab 🌟 */

document.addEventListener('DOMContentLoaded', () => {
    
    // 1. Galería de Módulo 1 (Pestañas LSTM)
    const tabButtons = document.querySelectorAll('.gallery-tab-btn');
    const gallerySlides = document.querySelectorAll('.gallery-slide');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remover estado activo de botones
            tabButtons.forEach(b => b.classList.remove('active'));
            // Agregar activo al seleccionado
            btn.classList.add('active');
            
            // Obtener el destino del botón
            const targetDest = btn.getAttribute('data-target');
            
            // Mostrar y ocultar slides correspondientes
            gallerySlides.forEach(slide => {
                const slideDest = slide.getAttribute('data-slide');
                if (slideDest === targetDest) {
                    slide.classList.add('active');
                } else {
                    slide.classList.remove('active');
                }
            });
        });
    });
    
    // 2. Active Sidebar Links on Scroll (Scroll Spy)
    const sections = document.querySelectorAll('.report-section');
    const navItems = document.querySelectorAll('.sidebar-item');
    
    window.addEventListener('scroll', () => {
        let currentSectionId = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            // Se activa cuando la sección cubre al menos la mitad superior de la pantalla
            if (window.scrollY >= (sectionTop - 150)) {
                currentSectionId = section.getAttribute('id');
            }
        });
        
        navItems.forEach(item => {
            item.classList.remove('active');
            const linkHref = item.querySelector('a').getAttribute('href');
            if (linkHref === `#${currentSectionId}`) {
                item.classList.add('active');
            }
        });
    });
    
    // 3. Añadir animaciones de entrada progresivas al hacer scroll
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };
    
    const elementObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active-animation');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Podemos añadir clases dinámicas si lo deseamos para animar elementos al aparecer en pantalla
    const animatedCards = document.querySelectorAll('.premium-card');
    animatedCards.forEach(card => {
        card.style.opacity = '1'; // El CSS se encarga de hovers
    });
});
