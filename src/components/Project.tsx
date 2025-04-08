import React from 'react'
import './Project.css'

import DescriptionIcon from '@mui/icons-material/Description';
import GitHubIcon from '@mui/icons-material/GitHub';


const Project = () => {
    return (
        <div className='project'>
            <div className='project__hero'>
                <div className='project__title'>
                    <p className='google-sans-semi-bold'>SyncSDE: A Probabilistic Framework for Diffusion Synchronization</p>
                </div>
                <div className='project__authorInfo'>
                    <div className='project__authors'>
                        <div className='project__author'><p className='google-sans-light'><a href='https://hjl1013.github.io/' target='_blank' rel='noopener noreferrer'>Hyunjun Lee</a></p>*</div>
                        <div className='project__author'><p className='google-sans-light'><a href='https://www.linkedin.com/in/philip21' target='_blank' rel='noopener noreferrer'>Hyunsoo Lee</a></p>*</div>
                        <div className='project__author'><p className='google-sans-light'><a href='https://jellyheadandrew.github.io/' target='_blank' rel='noopener noreferrer'>Sookwan Han</a></p>&dagger;</div>
                    </div>
                    <div className='project__affilations'>
                        <p className='project__affiliation google-sans-light'>ECE, Seoul National University</p>
                        <p className='project__affiliation google-sans-light'>Republic of Korea Air Force</p>
                    </div>
                    <div className='project__conference'>
                        <p className='project__conference-name google-sans-light'>CVPR 2025</p>
                    </div>
                    <div className='project__contributions'>
                        <p className='google-sans-light'>
                            * indicates equal contribution, &dagger; indicates project lead
                        </p>
                    </div>
                </div>
                <div className='project__links'>
                    <a href='https://arxiv.org/abs/2503.21555' target='_blank' rel='noopener noreferrer'>
                        <div className='project__link'>
                            <DescriptionIcon />
                            <p className='project__linkName google-sans-medium'>Paper</p>
                        </div>
                    </a>
                    <a href='https://github.com/hjl1013/SyncSDE' target='_blank' rel='noopener noreferrer'>
                        <div className='project__link'>
                            <GitHubIcon />
                            <p className='project__linkName google-sans-medium'>Code</p>
                        </div>
                    </a>
                </div>
            </div>

            <div className='project__body'>
                <div className='project__teaser project__content'>
                    <img src='./figures/CVPR2025_Qualitative_Teaser_Crop.jpg' width='100%' height='100%'/>
                    <p className='project__teaserText google-sans-light'>
                        SyncSDE analyzes diffusion synchronization to identify where the correlation strategies should be focused, 
                        enabling coherent and high-quality results across diverse collaborative generation tasks.
                    </p>
                </div>
                <div className='project__abstractContainer grey-background'>
                    <div className='project__abstract project__content'>
                        <p className='project__bodyTitle google-sans-medium'>Abstract</p>
                        <p className='project__abstractText google-sans-light'>
                            There have been many attempts to leverage multiple diffusion models for collaborative generation, 
                            extending beyond the original domain. A prominent approach involves synchronizing multiple diffusion 
                            trajectories by mixing the estimated scores to artificially correlate the generation processes. However, 
                            existing methods rely on naive heuristics, such as averaging, without considering task specificity. 
                            These approaches do not clarify why such methods work and often fail when a heuristic suitable for one 
                            task is blindly applied to others. In this paper, we present a probabilistic framework for analyzing 
                            why diffusion synchronization works and reveal where heuristics should be focused—modeling correlations 
                            between multiple trajectories and adapting them to each specific task. We further identify optimal correlation 
                            models per task, achieving better results than previous approaches that apply a single heuristic across 
                            all tasks without justification.
                        </p>
                    </div>
                </div>
                <div className='project__experiments project__content'>
                    <div className='project__mask project__experiment'>
                        <p className='project__bodyTitle google-sans-medium'>Mask-based Text-to-Image generation</p>
                        <div className='project__maskBody'>
                            <img src='./figures/CVPR2025_Qualitative_Mask_Generation.png' width='100%' height='100%'/>
                            <p className='google-sans-light'>
                                We first show that the mask-based text-to-image generation is a special case of diffusion synchronization.
                            </p>
                        </div>
                    </div>
                    <div className='project__mask project__experiment'>
                        <p className='project__bodyTitle google-sans-medium'>Text-driven real image editing</p>
                        <div className='project__editingBody'>
                            <p className='google-sans-light'>
                                We first show that the mask-based text-to-image generation is a special case of diffusion synchronization.
                            </p>
                            <img src='./figures/CVPR2025_Qualitative_Image_Editing_Crop.png' width='100%' height='100%'/>
                        </div>
                    </div>
                    <div className='project__mask project__experiment'>
                        <p className='project__bodyTitle google-sans-medium'>Wide image generation</p>
                        <div className='project__wideBody'>
                            <img src='./figures/CVPR2025_Qualitative_Wide_Image_Crop.png' width='100%' height='100%'/>
                            <p className='google-sans-light'>
                                We first show that the mask-based text-to-image generation is a special case of diffusion synchronization.
                            </p>
                        </div>
                    </div>
                    <div className='project__mask project__experiment'>
                        <p className='project__bodyTitle google-sans-medium'>Ambiguous image generation</p>
                        <div className='project__ambiguousBody'>
                            <p className='google-sans-light'>
                                We first show that the mask-based text-to-image generation is a special case of diffusion synchronization.
                            </p>
                            <img src='./figures/CVPR2025_Qualitative_Ambiguous_Images.png' width='100%' height='100%'/>
                        </div>
                    </div>
                    <div className='project__mask project__experiment'>
                        <p className='project__bodyTitle google-sans-medium'>3D mesh texturing</p>
                        <div className='project__meshBody'>
                            <img src='./figures/CVPR2025_Qualitative_Mesh_Texturing.png' width='100%' height='100%'/>
                            <p className='google-sans-light'>
                                We first show that the mask-based text-to-image generation is a special case of diffusion synchronization.
                            </p>
                        </div>
                    </div>
                    <div className='project__mask project__experiment'>
                        <p className='project__bodyTitle google-sans-medium'>Long-horizon motion generation</p>
                        <div className='project__motionBody'>
                            <p className='google-sans-light'>
                                We first show that the mask-based text-to-image generation is a special case of diffusion synchronization.
                            </p>
                            <img src='./figures/CVPR2025_Qualizative_Motion_Generation.png' width='100%' height='100%'/>
                        </div>
                    </div>
                </div>

                <div className='project__bibtex project__content'>
                    <p className='project__bodyTitle google-sans-medium'>BibTeX</p>
                    <pre className='google-sans-light grey-background'>
                        <code>
                            {`@article{lee2025syncsde,\n\ttitle={SyncSDE: A Probabilistic Framework for Diffusion Synchronization},\n\tauthor={Lee, Hyunjun and Lee, Hyunsoo and Han, Sookwan},\n\tbooktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},\n\tyear={2025}\n}`}
                        </code>
                    </pre>
                </div>
            </div>
        </div>
    )
}

export default Project